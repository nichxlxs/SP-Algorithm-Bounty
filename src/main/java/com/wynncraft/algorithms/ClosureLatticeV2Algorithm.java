package com.wynncraft.algorithms;

import com.wynncraft.core.WynnPlayer;
import com.wynncraft.core.interfaces.IAlgorithm;
import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.Information;
import com.wynncraft.enums.SkillPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * SWAR evolution of Closure Lattice: identical search semantics to V1 (same
 * forced-closure theory, greedy certificate, exact mask DFS with duplicate
 * canonicalization, admissible bound and node budget - see
 * {@link ClosureLatticeAlgorithm}), with the hot path operating on all five
 * skills packed into one long (12-bit biased lanes, guard-bit comparisons),
 * so requirement/invariant checks cost ~3 ops instead of ~15.
 *
 * Packing is only sound while every packed lane stays within [0, 2047]
 * (values in [-1023, 1023] after the +1024 bias). A per-call domain guard
 * verifies that from the actual input (per-lane |base| + sum |bonus| and
 * max requirement + max bonus); anything outside delegates to the scalar
 * V1 solver, so adversarial magnitudes lose speed, never correctness.
 * Real game data sits two orders of magnitude inside the domain.
 *
 * No state is carried between calls: scratch buffers are reused for memory
 * only and every cell read is rewritten each run. No equipment identity or
 * player caching of any kind.
 */
@Information(name = "Closure Lattice", version = 2, authors = {"claude"})
public class ClosureLatticeV2Algorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();
    private static final int S = 5;
    private static final int VISITED_BITSET_LIMIT = 20;
    private static final int EXACT_LIMIT = 62;
    private static final int NODE_BUDGET = 1 << 16;

    private static final long BIAS_5 = 0x0400_4004_0040_0400L;   // 1024 in each lane
    private static final long GUARD = 0x0800_8008_0080_0800L;    // bit 11 of each lane

    /** Fallback exact solver for inputs outside the packable domain. */
    private final ClosureLatticeAlgorithm scalar = new ClosureLatticeAlgorithm();

    // Per-item packed data.
    private long[] reqPk = new long[0];
    private long[] bonPk = new long[0];
    private int[][] bonRef = new int[0][];
    private boolean[] hasNeg = new boolean[0];
    private int[] itemScore = new int[0];
    private boolean[] valid = new boolean[0];
    private final int[] base = new int[S];
    private final int[] bonusTotal = new int[S];

    // Classification.
    private int[] forcedIdx = new int[0];
    private int[] forcedKeys = new int[0];
    private int[] branchIdx = new int[0];
    private int[] branchKeys = new int[0];
    private long[] bReqPk = new long[0];    // branch position -> packed requirement
    private long[] bBonPk = new long[0];
    private long[] bRpbPk = new long[0];    // packed max(0, req+bonus-biased) sentinel form
    private int[] bScore = new int[0];
    private long[] dupPred = new long[0];

    // Search scratch.
    private boolean[] equippedItem = new boolean[0];
    private int[] closureOrder = new int[0];
    private long[] statsPkStack = new long[0];   // per depth
    private long[] needPkStack = new long[0];
    private int[] remPosStack = new int[0];      // per depth * 5 (scalar; bound only)
    private int[] posScoreStack = new int[0];
    private long[] visited = new long[0];
    private HashSet<Long> visitedLarge;

    private int n;
    private int forcedCount;
    private int branchCount;
    private int closureSize;
    private int bestCount;
    private int bestWeight;
    private long bestMask;
    private boolean solved;
    private int nodesLeft;

    @Override
    public void clearCache() {
        scalar.clearCache();
    }

    private static long pack(int a, int b, int c, int d, int e) {
        return (long) (a + 1024)
            | ((long) (b + 1024)) << 12
            | ((long) (c + 1024)) << 24
            | ((long) (d + 1024)) << 36
            | ((long) (e + 1024)) << 48;
    }

    /** Per-lane a >= b for same-biased packed values with lanes in [0, 2047]. */
    private static boolean ge5(long a, long b) {
        return (((a | GUARD) - b) & GUARD) == GUARD;
    }

    /** Per-lane maximum of two packed values. */
    private static long max5(long a, long b) {
        long gt = ((a | GUARD) - b) & GUARD;
        long ones = gt >>> 11;
        long mask = gt | (gt - ones);
        return (a & mask) | (b & ~mask);
    }

    private static int lane(long pk, int s) {
        return (int) ((pk >>> (12 * s)) & 0xFFF) - 1024;
    }

    @Override
    public Result run(WynnPlayer player) {
        List<IEquipment> equipment = player.equipment();
        int count = equipment.size();
        if (count == 0) {
            return new Result(new ArrayList<>(0), new ArrayList<>(0));
        }

        this.n = count;
        ensureCapacity(count);
        for (int s = 0; s < S; s++) {
            base[s] = player.allocated(SKILL_POINTS[s]);
        }

        // Pass A: pack items, gather the reducible-skill mask, and check the
        // packing domain (per-lane |base| + sum|bonus| <= 1023 keeps every
        // reachable stat lane in range; maxReq + maxPosBonus <= 1023 keeps
        // requirement and req+bonus lanes in range).
        int riskyMask = 0;
        int maxReq = 0;
        int maxPosBonus = 0;
        long absLane0 = Math.abs(base[0]);
        long absLane1 = Math.abs(base[1]);
        long absLane2 = Math.abs(base[2]);
        long absLane3 = Math.abs(base[3]);
        long absLane4 = Math.abs(base[4]);
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            bonRef[i] = b;
            boolean neg = item.hasNegativeBonus();
            hasNeg[i] = neg;
            int r0 = Math.max(r[0], 0);
            int r1 = Math.max(r[1], 0);
            int r2 = Math.max(r[2], 0);
            int r3 = Math.max(r[3], 0);
            int r4 = Math.max(r[4], 0);
            // Requirement lanes with r <= 0 pack as biased-0: always satisfied.
            reqPk[i] = (long) (r0 == 0 ? 0 : r0 + 1024)
                | ((long) (r1 == 0 ? 0 : r1 + 1024)) << 12
                | ((long) (r2 == 0 ? 0 : r2 + 1024)) << 24
                | ((long) (r3 == 0 ? 0 : r3 + 1024)) << 36
                | ((long) (r4 == 0 ? 0 : r4 + 1024)) << 48;
            bonPk[i] = pack(b[0], b[1], b[2], b[3], b[4]);
            itemScore[i] = b[0] + b[1] + b[2] + b[3] + b[4];
            maxReq = Math.max(maxReq, Math.max(Math.max(r0, r1), Math.max(Math.max(r2, r3), r4)));
            for (int s = 0; s < S; s++) {
                int bv = b[s];
                if (bv < 0) {
                    riskyMask |= 1 << s;
                } else if (bv > maxPosBonus) {
                    maxPosBonus = bv;
                }
            }
            absLane0 += Math.abs(b[0]);
            absLane1 += Math.abs(b[1]);
            absLane2 += Math.abs(b[2]);
            absLane3 += Math.abs(b[3]);
            absLane4 += Math.abs(b[4]);
        }

        long maxAbs = Math.max(Math.max(absLane0, absLane1),
            Math.max(Math.max(absLane2, absLane3), absLane4));
        if (maxAbs > 1023 || maxReq + maxPosBonus > 1023 || maxReq > 1023) {
            return scalar.run(player);
        }

        if (riskyMask == 0) {
            pureClosure();
        } else {
            solve(riskyMask);
        }

        List<IEquipment> validList = new ArrayList<>(count);
        List<IEquipment> invalidList = new ArrayList<>(count);
        for (int s = 0; s < S; s++) {
            bonusTotal[s] = 0;
        }
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (valid[i]) {
                validList.add(item);
                int[] b = bonRef[i];
                for (int s = 0; s < S; s++) {
                    bonusTotal[s] += b[s];
                }
            } else {
                invalidList.add(item);
            }
        }
        if (!validList.isEmpty()) {
            player.modify(bonusTotal, true);
        }
        return new Result(validList, invalidList);
    }

    private void ensureCapacity(int items) {
        if (valid.length >= items) {
            return;
        }
        int cap = Math.max(items, valid.length * 2 + 8);
        reqPk = new long[cap];
        bonPk = new long[cap];
        bonRef = new int[cap][];
        hasNeg = new boolean[cap];
        itemScore = new int[cap];
        valid = new boolean[cap];
        forcedIdx = new int[cap];
        forcedKeys = new int[cap];
        branchIdx = new int[cap];
        branchKeys = new int[cap];
        bReqPk = new long[cap];
        bBonPk = new long[cap];
        bRpbPk = new long[cap];
        bScore = new int[cap];
        dupPred = new long[cap];
        equippedItem = new boolean[cap];
        closureOrder = new int[cap];
        statsPkStack = new long[cap + 2];
        needPkStack = new long[cap + 2];
        remPosStack = new int[(cap + 2) * S];
        posScoreStack = new int[cap + 2];
    }

    private void pureClosure() {
        Arrays.fill(valid, 0, n, false);
        long stats = pack(base[0], base[1], base[2], base[3], base[4]);
        boolean progress = true;
        while (progress) {
            progress = false;
            for (int i = 0; i < n; i++) {
                if (valid[i] || !ge5(stats, reqPk[i])) {
                    continue;
                }
                stats += bonPk[i] - BIAS_5;
                valid[i] = true;
                progress = true;
            }
        }
    }

    private void solve(int riskyMask) {
        classify(riskyMask);

        Arrays.fill(equippedItem, 0, n, false);
        int greedyCount = runGreedy();

        if (greedyCount == n) {
            Arrays.fill(valid, 0, n, true);
            return;
        }

        if (branchCount == 0) {
            System.arraycopy(equippedItem, 0, valid, 0, n);
            return;
        }

        if (branchCount > EXACT_LIMIT) {
            fallbackWithRepair(greedyCount);
            return;
        }

        long greedyMask = greedyBranchMask();
        bestCount = greedyCount;
        bestWeight = pathWeight(greedyMask);
        bestMask = greedyMask;
        solved = false;
        nodesLeft = NODE_BUDGET;

        prepareVisited();
        Arrays.fill(equippedItem, 0, n, false);
        initRemaining();
        statsPkStack[0] = pack(base[0], base[1], base[2], base[3], base[4]);
        needPkStack[0] = 0L;
        closureSize = 0;
        int rootCount = runClosure(0);
        int rootWeight = closureScoreFrom(0);
        dfs(0L, 0, rootCount, rootWeight);
        visitedLarge = null;

        reconstruct(bestMask);
    }

    private void classify(int riskyMask) {
        // Requirement-lane mask per item is derivable from reqPk (nonzero lanes).
        forcedCount = 0;
        branchCount = 0;
        for (int i = 0; i < n; i++) {
            long rp = reqPk[i];
            int reqLanes = 0;
            int reqSum = 0;
            for (int s = 0; s < S; s++) {
                int v = (int) ((rp >>> (12 * s)) & 0xFFF);
                if (v != 0) {
                    reqLanes |= 1 << s;
                    reqSum += v - 1024;
                }
            }
            if (!hasNeg[i] && (reqLanes & riskyMask) == 0) {
                forcedIdx[forcedCount] = i;
                forcedKeys[forcedCount] = reqSum;
                forcedCount++;
            } else {
                int p = branchCount++;
                branchIdx[p] = i;
                bReqPk[p] = rp;
                bBonPk[p] = bonPk[i];
                bScore[p] = itemScore[i];
                // rpb: lanes with req > 0 pack (req + bonus + 1024); others biased-0.
                int[] b = bonRef[i];
                long rpb = 0L;
                int rpbSum = 0;
                for (int s = 0; s < S; s++) {
                    int v = (int) ((rp >>> (12 * s)) & 0xFFF);
                    if (v != 0) {
                        int rb = (v - 1024) + b[s];
                        rpb |= ((long) (rb + 1024)) << (12 * s);
                        rpbSum += rb;
                    }
                }
                bRpbPk[p] = rpb;
                branchKeys[p] = hasNeg[i] ? (1 << 30) - rpbSum : reqSum;
            }
        }
        insertionSort(forcedIdx, forcedKeys, forcedCount);
        sortBranch();

        for (int p = 0; p < branchCount && p < 64; p++) {
            long mask = 0L;
            for (int q = 0; q < p; q++) {
                if (bReqPk[q] == bReqPk[p] && bBonPk[q] == bBonPk[p]) {
                    mask |= 1L << q;
                }
            }
            dupPred[p] = mask;
        }
    }

    private void sortBranch() {
        for (int i = 1; i < branchCount; i++) {
            int key = branchKeys[i];
            int idx = branchIdx[i];
            if (branchKeys[i - 1] < key || (branchKeys[i - 1] == key && branchIdx[i - 1] < idx)) {
                continue;
            }
            long rq = bReqPk[i];
            long bo = bBonPk[i];
            long rb = bRpbPk[i];
            int sc = bScore[i];
            int j = i - 1;
            while (j >= 0 && (branchKeys[j] > key || (branchKeys[j] == key && branchIdx[j] > idx))) {
                branchKeys[j + 1] = branchKeys[j];
                branchIdx[j + 1] = branchIdx[j];
                bReqPk[j + 1] = bReqPk[j];
                bBonPk[j + 1] = bBonPk[j];
                bRpbPk[j + 1] = bRpbPk[j];
                bScore[j + 1] = bScore[j];
                j--;
            }
            branchKeys[j + 1] = key;
            branchIdx[j + 1] = idx;
            bReqPk[j + 1] = rq;
            bBonPk[j + 1] = bo;
            bRpbPk[j + 1] = rb;
            bScore[j + 1] = sc;
        }
    }

    private static void insertionSort(int[] indices, int[] keys, int length) {
        for (int i = 1; i < length; i++) {
            int index = indices[i];
            int key = keys[i];
            int j = i - 1;
            while (j >= 0 && (keys[j] > key || (keys[j] == key && indices[j] > index))) {
                indices[j + 1] = indices[j];
                keys[j + 1] = keys[j];
                j--;
            }
            indices[j + 1] = index;
            keys[j + 1] = key;
        }
    }

    private int runGreedy() {
        statsPkStack[0] = pack(base[0], base[1], base[2], base[3], base[4]);
        needPkStack[0] = 0L;
        closureSize = 0;
        int count = runClosure(0);
        boolean progress = branchCount > 0;
        while (progress) {
            progress = false;
            for (int p = 0; p < branchCount; p++) {
                int i = branchIdx[p];
                if (equippedItem[i]) {
                    continue;
                }
                long stats = statsPkStack[0];
                if (!ge5(stats, bReqPk[p])) {
                    continue;
                }
                long newStats = stats + bBonPk[p] - BIAS_5;
                long newNeed = max5(needPkStack[0], bRpbPk[p]);
                if (!ge5(newStats, newNeed)) {
                    continue;
                }
                statsPkStack[0] = newStats;
                needPkStack[0] = newNeed;
                equippedItem[i] = true;
                count++;
                count += runClosure(0);
                progress = true;
            }
        }
        return count;
    }

    private long greedyBranchMask() {
        long mask = 0L;
        for (int p = 0; p < branchCount && p < 64; p++) {
            if (equippedItem[branchIdx[p]]) {
                mask |= 1L << p;
            }
        }
        return mask;
    }

    private void fallbackWithRepair(int greedyCount) {
        // Beyond exact reach; greedy answer improved by leave-one-out repair.
        System.arraycopy(equippedItem, 0, valid, 0, n);
        int bestC = greedyCount;
        int bestW = equippedScore();
        int attempts = 0;
        for (int p = 0; p < branchCount && attempts < 64; p++) {
            int i = branchIdx[p];
            if (!valid[i] || !hasNeg[i]) {
                continue;
            }
            attempts++;
            Arrays.fill(equippedItem, 0, n, false);
            equippedItem[i] = true;             // pre-marking excludes it from greedy
            int count = runGreedy() ;
            equippedItem[i] = false;
            int weight = equippedScore();
            if (count > bestC || (count == bestC && weight > bestW)) {
                bestC = count;
                bestW = weight;
                System.arraycopy(equippedItem, 0, valid, 0, n);
            }
        }
    }

    private int equippedScore() {
        int weight = 0;
        for (int i = 0; i < n; i++) {
            if (equippedItem[i]) {
                weight += itemScore[i];
            }
        }
        return weight;
    }

    private void initRemaining() {
        for (int s = 0; s < S; s++) {
            remPosStack[s] = 0;
        }
        int posScore = 0;
        for (int i = 0; i < n; i++) {
            int[] b = bonRef[i];
            int sc = 0;
            for (int s = 0; s < S; s++) {
                int bv = b[s];
                sc += bv;
                if (bv > 0) {
                    remPosStack[s] += bv;
                }
            }
            if (sc > 0) {
                posScore += sc;
            }
        }
        posScoreStack[0] = posScore;
    }

    private void prepareVisited() {
        if (branchCount <= VISITED_BITSET_LIMIT) {
            int words = Math.max(1, (1 << branchCount) >>> 6);
            if (visited.length < words) {
                visited = new long[words];
            } else {
                Arrays.fill(visited, 0, words, 0L);
            }
            visitedLarge = null;
        } else {
            visitedLarge = new HashSet<>();
        }
    }

    private boolean visit(long mask) {
        if (visitedLarge != null) {
            return visitedLarge.add(mask);
        }
        int word = (int) (mask >>> 6);
        long bit = 1L << (mask & 63);
        if ((visited[word] & bit) != 0) {
            return false;
        }
        visited[word] |= bit;
        return true;
    }

    private int runClosure(int depth) {
        long stats = statsPkStack[depth];
        int d = depth * S;
        int added = 0;
        boolean progress = true;
        while (progress) {
            progress = false;
            for (int p = 0; p < forcedCount; p++) {
                int i = forcedIdx[p];
                if (equippedItem[i] || !ge5(stats, reqPk[i])) {
                    continue;
                }
                stats += bonPk[i] - BIAS_5;
                int[] b = bonRef[i];
                for (int s = 0; s < S; s++) {
                    if (b[s] > 0) {
                        remPosStack[d + s] -= b[s];
                    }
                }
                if (itemScore[i] > 0) {
                    posScoreStack[depth] -= itemScore[i];
                }
                equippedItem[i] = true;
                closureOrder[closureSize++] = i;
                added++;
                progress = true;
            }
        }
        statsPkStack[depth] = stats;
        return added;
    }

    private int pathWeight(long branchMask) {
        int weight = 0;
        for (long m = branchMask; m != 0; m &= m - 1) {
            weight += bScore[Long.numberOfTrailingZeros(m)];
        }
        for (int k = 0; k < closureSize; k++) {
            weight += itemScore[closureOrder[k]];
        }
        return weight;
    }

    private int closureScoreFrom(int start) {
        int weight = 0;
        for (int k = start; k < closureSize; k++) {
            weight += itemScore[closureOrder[k]];
        }
        return weight;
    }

    private void dfs(long mask, int depth, int count, int weight) {
        if (count > bestCount || (count == bestCount && weight > bestWeight)) {
            bestCount = count;
            bestWeight = weight;
            bestMask = mask;
            if (count == n) {
                solved = true;
                return;
            }
        }
        if (--nodesLeft < 0) {
            return;
        }

        long stats = statsPkStack[depth];
        int d = depth * S;
        // Admissible bound (scalar lanes; runs once per node, not per child).
        long reach = 0L;
        for (int s = 0; s < S; s++) {
            int v = lane(stats, s) + remPosStack[d + s];
            reach |= ((long) (Math.min(v, 1023) + 1024)) << (12 * s);
        }
        int countBound = count;
        for (int p = 0; p < branchCount; p++) {
            if ((mask & (1L << p)) == 0 && ge5(reach, bReqPk[p])) {
                countBound++;
            }
        }
        for (int p = 0; p < forcedCount; p++) {
            int i = forcedIdx[p];
            if (!equippedItem[i] && ge5(reach, reqPk[i])) {
                countBound++;
            }
        }
        if (countBound < bestCount
            || (countBound == bestCount && weight + posScoreStack[depth] <= bestWeight)) {
            return;
        }

        for (int p = 0; p < branchCount; p++) {
            long bit = 1L << p;
            if ((mask & bit) != 0 || (mask & dupPred[p]) != dupPred[p]) {
                continue;
            }
            if (!ge5(stats, bReqPk[p])) {
                continue;
            }
            long newStats = stats + bBonPk[p] - BIAS_5;
            long newNeed = max5(needPkStack[depth], bRpbPk[p]);
            if (!ge5(newStats, newNeed)) {
                continue;
            }
            long newMask = mask | bit;
            if (!visit(newMask)) {
                continue;
            }
            statsPkStack[depth + 1] = newStats;
            needPkStack[depth + 1] = newNeed;
            int nd = d + S;
            int[] b = bonRef[branchIdx[p]];
            for (int s = 0; s < S; s++) {
                remPosStack[nd + s] = remPosStack[d + s] - Math.max(b[s], 0);
            }
            posScoreStack[depth + 1] = posScoreStack[depth] - Math.max(bScore[p], 0);
            int savedClosure = closureSize;
            int added = runClosure(depth + 1);
            int addedWeight = bScore[p] + closureScoreFrom(savedClosure);
            dfs(newMask, depth + 1, count + 1 + added, weight + addedWeight);
            if (solved) {
                return;
            }
            for (int k = savedClosure; k < closureSize; k++) {
                equippedItem[closureOrder[k]] = false;
            }
            closureSize = savedClosure;
        }
    }

    private void reconstruct(long branchMask) {
        Arrays.fill(valid, 0, n, false);
        Arrays.fill(equippedItem, 0, n, false);
        long stats = pack(base[0], base[1], base[2], base[3], base[4]);
        for (long m = branchMask; m != 0; m &= m - 1) {
            int p = Long.numberOfTrailingZeros(m);
            valid[branchIdx[p]] = true;
            stats += bBonPk[p] - BIAS_5;
        }
        statsPkStack[0] = stats;
        closureSize = 0;
        runClosure(0);
        for (int k = 0; k < closureSize; k++) {
            valid[closureOrder[k]] = true;
        }
    }

}
