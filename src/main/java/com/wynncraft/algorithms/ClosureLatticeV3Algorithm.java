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
 * Remainder-set evolution of Closure Lattice. Search semantics are identical
 * to V1 (forced-closure theory, greedy certificate, exact mask DFS with
 * duplicate canonicalization, admissible bound, node budget, no caps), with
 * the work restructured so per-call cost concentrates on the few items that
 * actually need deciding:
 *
 *  - One extraction pass equips every no-requirement, non-negative item
 *    immediately (unconditionally optimal) without packing or copying it.
 *  - Requirement-bearing items on non-reducible lanes ("forced") and
 *    everything else ("branch") are SWAR-packed - only these survivors, a
 *    small minority of a real build, are ever touched again.
 *  - Pending forced items live in a compact swap-remove list with LIFO undo,
 *    so closure scans inside the greedy loop and at every DFS node touch
 *    only still-unequipped items instead of the whole forced set.
 *
 * All five skill lanes are packed into one long (12-bit biased lanes,
 * guard-bit compares). A per-call domain guard (per-lane |base| + sum|bonus|
 * <= 1023, max requirement + max positive bonus <= 1023) delegates anything
 * unpackable to the scalar V1 solver: adversarial magnitudes lose speed,
 * never correctness. No state is carried between calls and no equipment or
 * player caching of any kind is used.
 */
@Information(name = "Closure Lattice", version = 3, authors = {"claude"})
public class ClosureLatticeV3Algorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();
    private static final int S = 5;
    private static final int VISITED_BITSET_LIMIT = 20;
    private static final int EXACT_LIMIT = 62;
    private static final int NODE_BUDGET = 1 << 16;

    private static final long BIAS_5 = 0x0400_4004_0040_0400L;
    private static final long GUARD = 0x0800_8008_0080_0800L;

    /** Fallback exact solver for inputs outside the packable domain. */
    private final ClosureLatticeAlgorithm scalar = new ClosureLatticeAlgorithm();

    // Per-item minimal data.
    private int[][] bonRef = new int[0][];
    private boolean[] valid = new boolean[0];
    private boolean[] negFlag = new boolean[0];
    private int[] wlIdx = new int[0];
    private int[][] wlReq = new int[0][];
    private final int[] base = new int[S];
    private final int[] bonusTotal = new int[S];

    // Forced remainder (item indices + packed data), compact pending list.
    private int[] forcedItem = new int[0];
    private long[] fReqPk = new long[0];
    private long[] fBonPk = new long[0];
    private int[] fScore = new int[0];
    private int[] fKeys = new int[0];
    private int[] pending = new int[0];        // indices into forced arrays
    private int[] pendingPos = new int[0];     // forced index -> current slot in pending
    private int pendingCount;

    // Branch remainder (position-indexed packed data).
    private int[] branchItem = new int[0];
    private long[] bReqPk = new long[0];
    private long[] bBonPk = new long[0];
    private long[] bRpbPk = new long[0];
    private int[] bScore = new int[0];
    private int[] bKeys = new int[0];
    private long[] dupPred = new long[0];

    // Search scratch.
    private boolean[] equippedForced = new boolean[0]; // by forced index
    private int[] closureLog = new int[0];             // forced indices, LIFO undo
    private long[] statsPkStack = new long[0];
    private long[] needPkStack = new long[0];
    private int[] remPosStack = new int[0];
    private int[] posScoreStack = new int[0];
    private long[] visited = new long[0];
    private HashSet<Long> visitedLarge;

    private int n;
    private int forcedCount;
    private int branchCount;
    private int closureSize;
    private int extractedCount;
    private int extractedScore;
    private long extractedStatsPk;
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

    private static boolean ge5(long a, long b) {
        return (((a | GUARD) - b) & GUARD) == GUARD;
    }

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

        // Pass 0: negative-bonus pre-scan (one precomputed-flag call per item).
        boolean anyNeg = false;
        for (int i = 0; i < count; i++) {
            boolean neg = equipment.get(i).hasNegativeBonus();
            negFlag[i] = neg;
            anyNeg |= neg;
        }

        if (!anyNeg) {
            // No item reduces any skill: feasibility is a monotone closure.
            // Scalar, no packing, no domain guard needed; a compact worklist
            // keeps fixpoint passes to the still-undecided items only.
            int s0 = base[0];
            int s1 = base[1];
            int s2 = base[2];
            int s3 = base[3];
            int s4 = base[4];
            int wl = 0;
            for (int i = 0; i < count; i++) {
                IEquipment item = equipment.get(i);
                int[] r = item.requirements();
                if ((r[0] <= 0 || s0 >= r[0]) && (r[1] <= 0 || s1 >= r[1])
                    && (r[2] <= 0 || s2 >= r[2]) && (r[3] <= 0 || s3 >= r[3])
                    && (r[4] <= 0 || s4 >= r[4])) {
                    int[] b = item.bonuses();
                    s0 += b[0];
                    s1 += b[1];
                    s2 += b[2];
                    s3 += b[3];
                    s4 += b[4];
                    valid[i] = true;
                } else {
                    valid[i] = false;
                    wlIdx[wl] = i;
                    wlReq[wl] = r;
                    wl++;
                }
            }
            boolean progress = wl > 0;
            while (progress) {
                progress = false;
                for (int k = 0; k < wl; k++) {
                    int[] r = wlReq[k];
                    if ((r[0] > 0 && s0 < r[0]) || (r[1] > 0 && s1 < r[1])
                        || (r[2] > 0 && s2 < r[2]) || (r[3] > 0 && s3 < r[3])
                        || (r[4] > 0 && s4 < r[4])) {
                        continue;
                    }
                    int i = wlIdx[k];
                    int[] b = equipment.get(i).bonuses();
                    s0 += b[0];
                    s1 += b[1];
                    s2 += b[2];
                    s3 += b[3];
                    s4 += b[4];
                    valid[i] = true;
                    wl--;
                    wlIdx[k] = wlIdx[wl];
                    wlReq[k] = wlReq[wl];
                    k--;
                    progress = true;
                }
            }
            return buildResult(equipment, count, player);
        }

        // Pass 1: extraction + survivor collection in a single sweep. Extracted
        // items (no requirements, no negative bonus - unconditionally optimal)
        // cost a handful of scalar adds; every other cost, including the
        // domain-guard arithmetic, is confined to the few survivors.
        int riskyMask = 0;
        int ext0 = base[0];
        int ext1 = base[1];
        int ext2 = base[2];
        int ext3 = base[3];
        int ext4 = base[4];
        int survivors = 0;
        extractedCount = 0;
        extractedScore = 0;
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            boolean neg = negFlag[i];
            if (!neg && r[0] <= 0 && r[1] <= 0 && r[2] <= 0 && r[3] <= 0 && r[4] <= 0) {
                valid[i] = true;
                extractedCount++;
                extractedScore += b[0] + b[1] + b[2] + b[3] + b[4];
                ext0 += b[0];
                ext1 += b[1];
                ext2 += b[2];
                ext3 += b[3];
                ext4 += b[4];
                continue;
            }
            valid[i] = false;
            bonRef[i] = b;
            wlIdx[survivors] = i;
            wlReq[survivors] = r;
            survivors++;
            if (neg) {
                for (int s = 0; s < S; s++) {
                    if (b[s] < 0) {
                        riskyMask |= 1 << s;
                    }
                }
            }
        }

        // Pass 2: classification, packing and the domain guard - survivors only.
        // Cumulative lane bounds: |base| plus extracted positives (ext - base)
        // plus survivor |bonus| sums.
        int maxReq = 0;
        int maxPosBonus = 0;
        long abs0 = Math.abs(base[0]) + (ext0 - base[0]);
        long abs1 = Math.abs(base[1]) + (ext1 - base[1]);
        long abs2 = Math.abs(base[2]) + (ext2 - base[2]);
        long abs3 = Math.abs(base[3]) + (ext3 - base[3]);
        long abs4 = Math.abs(base[4]) + (ext4 - base[4]);
        forcedCount = 0;
        branchCount = 0;
        for (int k = 0; k < survivors; k++) {
            int i = wlIdx[k];
            int[] r = wlReq[k];
            int[] b = bonRef[i];
            abs0 += Math.abs(b[0]);
            abs1 += Math.abs(b[1]);
            abs2 += Math.abs(b[2]);
            abs3 += Math.abs(b[3]);
            abs4 += Math.abs(b[4]);
            int mx = Math.max(Math.max(b[0], b[1]), Math.max(Math.max(b[2], b[3]), b[4]));
            if (mx > maxPosBonus) {
                maxPosBonus = mx;
            }
            int r0 = Math.max(r[0], 0);
            int r1 = Math.max(r[1], 0);
            int r2 = Math.max(r[2], 0);
            int r3 = Math.max(r[3], 0);
            int r4 = Math.max(r[4], 0);
            int reqLanes = (r0 > 0 ? 1 : 0) | (r1 > 0 ? 2 : 0) | (r2 > 0 ? 4 : 0)
                | (r3 > 0 ? 8 : 0) | (r4 > 0 ? 16 : 0);
            int reqMax = Math.max(Math.max(r0, r1), Math.max(Math.max(r2, r3), r4));
            if (reqMax > maxReq) {
                maxReq = reqMax;
            }
            boolean neg = negFlag[i];
            int sc = b[0] + b[1] + b[2] + b[3] + b[4];
            long reqPk = (long) (r0 == 0 ? 0 : r0 + 1024)
                | ((long) (r1 == 0 ? 0 : r1 + 1024)) << 12
                | ((long) (r2 == 0 ? 0 : r2 + 1024)) << 24
                | ((long) (r3 == 0 ? 0 : r3 + 1024)) << 36
                | ((long) (r4 == 0 ? 0 : r4 + 1024)) << 48;
            long bonPk = pack(b[0], b[1], b[2], b[3], b[4]);
            if (!neg && (reqLanes & riskyMask) == 0) {
                int f = forcedCount++;
                forcedItem[f] = i;
                fReqPk[f] = reqPk;
                fBonPk[f] = bonPk;
                fScore[f] = sc;
                fKeys[f] = r0 + r1 + r2 + r3 + r4;
            } else {
                int p = branchCount++;
                branchItem[p] = i;
                bReqPk[p] = reqPk;
                bBonPk[p] = bonPk;
                bScore[p] = sc;
                long rpb = 0L;
                int rpbSum = 0;
                for (int s = 0; s < S; s++) {
                    int rv = s == 0 ? r0 : s == 1 ? r1 : s == 2 ? r2 : s == 3 ? r3 : r4;
                    if (rv > 0) {
                        int v = rv + b[s];
                        rpb |= ((long) (v + 1024)) << (12 * s);
                        rpbSum += v;
                    }
                }
                bRpbPk[p] = rpb;
                bKeys[p] = neg ? (1 << 30) - rpbSum : r0 + r1 + r2 + r3 + r4;
            }
        }

        long maxAbs = Math.max(Math.max(abs0, abs1), Math.max(Math.max(abs2, abs3), abs4));
        if (maxAbs > 1023 || maxReq + maxPosBonus > 1023 || maxReq > 1023
            || branchCount > EXACT_LIMIT) {
            // Outside the packable domain, or beyond exact subset search
            // (where V1 additionally applies leave-one-out greedy repair):
            // delegate wholesale to the scalar solver.
            return scalar.run(player);
        }

        extractedStatsPk = pack(ext0, ext1, ext2, ext3, ext4);
        solve();
        return buildResult(equipment, count, player);
    }

    private Result buildResult(List<IEquipment> equipment, int count, WynnPlayer player) {
        List<IEquipment> validList = new ArrayList<>(count);
        List<IEquipment> invalidList = new ArrayList<>(count);
        for (int s = 0; s < S; s++) {
            bonusTotal[s] = 0;
        }
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (valid[i]) {
                validList.add(item);
                int[] b = item.bonuses();
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
        bonRef = new int[cap][];
        valid = new boolean[cap];
        forcedItem = new int[cap];
        fReqPk = new long[cap];
        fBonPk = new long[cap];
        fScore = new int[cap];
        fKeys = new int[cap];
        pending = new int[cap];
        pendingPos = new int[cap];
        branchItem = new int[cap];
        bReqPk = new long[cap];
        bBonPk = new long[cap];
        bRpbPk = new long[cap];
        bScore = new int[cap];
        bKeys = new int[cap];
        dupPred = new long[cap];
        negFlag = new boolean[cap];
        wlIdx = new int[cap];
        wlReq = new int[cap][];
        equippedForced = new boolean[cap];
        closureLog = new int[cap];
        statsPkStack = new long[cap + 2];
        needPkStack = new long[cap + 2];
        remPosStack = new int[(cap + 2) * S];
        posScoreStack = new int[cap + 2];
    }

    private void solve() {
        if (forcedCount + branchCount == 0) {
            return; // extraction equipped everything relevant
        }
        // Sorting is a pass-count heuristic, not a correctness requirement;
        // for the small survivor sets of real builds the scan order is fine.
        if (forcedCount > 8) {
            sortForced();
        }
        if (branchCount > 8) {
            sortBranch();
        }

        // Greedy constructive attempt.
        resetPending();
        statsPkStack[0] = extractedStatsPk;
        needPkStack[0] = 0L;
        closureSize = 0;
        int count = extractedCount + runClosure(0);
        long greedyMask = 0L;
        if (branchCount > 0) {
            boolean progress = true;
            while (progress) {
                progress = false;
                for (int p = 0; p < branchCount; p++) {
                    if ((greedyMask & (1L << p)) != 0) {
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
                    greedyMask |= 1L << p;
                    count++;
                    count += runClosure(0);
                    progress = true;
                }
            }
        }

        if (count == n) {
            Arrays.fill(valid, 0, n, true);
            return;
        }

        if (branchCount == 0) {
            // Closure alone is exact.
            markClosure();
            return;
        }

        computeDupPred();
        bestCount = count;
        bestWeight = pathWeight(greedyMask);
        bestMask = greedyMask;
        solved = false;
        nodesLeft = NODE_BUDGET;

        prepareVisited();
        resetPending();
        initRemaining();
        statsPkStack[0] = extractedStatsPk;
        needPkStack[0] = 0L;
        closureSize = 0;
        int rootCount = extractedCount + runClosure(0);
        int rootWeight = extractedScore + closureScoreFrom(0);
        dfs(0L, 0, rootCount, rootWeight);
        visitedLarge = null;

        reconstruct(bestMask);
    }

    private void resetPending() {
        pendingCount = forcedCount;
        for (int f = 0; f < forcedCount; f++) {
            pending[f] = f;
            pendingPos[f] = f;
            equippedForced[f] = false;
        }
    }

    private void sortForced() {
        for (int i = 1; i < forcedCount; i++) {
            int key = fKeys[i];
            int item = forcedItem[i];
            if (fKeys[i - 1] <= key) {
                continue;
            }
            long rq = fReqPk[i];
            long bo = fBonPk[i];
            int sc = fScore[i];
            int j = i - 1;
            while (j >= 0 && fKeys[j] > key) {
                fKeys[j + 1] = fKeys[j];
                forcedItem[j + 1] = forcedItem[j];
                fReqPk[j + 1] = fReqPk[j];
                fBonPk[j + 1] = fBonPk[j];
                fScore[j + 1] = fScore[j];
                j--;
            }
            fKeys[j + 1] = key;
            forcedItem[j + 1] = item;
            fReqPk[j + 1] = rq;
            fBonPk[j + 1] = bo;
            fScore[j + 1] = sc;
        }
    }

    private void sortBranch() {
        for (int i = 1; i < branchCount; i++) {
            int key = bKeys[i];
            int item = branchItem[i];
            if (bKeys[i - 1] < key || (bKeys[i - 1] == key && branchItem[i - 1] < item)) {
                continue;
            }
            long rq = bReqPk[i];
            long bo = bBonPk[i];
            long rb = bRpbPk[i];
            int sc = bScore[i];
            int j = i - 1;
            while (j >= 0 && (bKeys[j] > key || (bKeys[j] == key && branchItem[j] > item))) {
                bKeys[j + 1] = bKeys[j];
                branchItem[j + 1] = branchItem[j];
                bReqPk[j + 1] = bReqPk[j];
                bBonPk[j + 1] = bBonPk[j];
                bRpbPk[j + 1] = bRpbPk[j];
                bScore[j + 1] = bScore[j];
                j--;
            }
            bKeys[j + 1] = key;
            branchItem[j + 1] = item;
            bReqPk[j + 1] = rq;
            bBonPk[j + 1] = bo;
            bRpbPk[j + 1] = rb;
            bScore[j + 1] = sc;
        }
    }

    private void computeDupPred() {
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

    /**
     * Closure over the pending forced list only. Swap-removes equipped items;
     * the closure log lets the DFS undo in exact reverse order.
     */
    private int runClosure(int depth) {
        long stats = statsPkStack[depth];
        int d = depth * S;
        int added = 0;
        boolean progress = true;
        while (progress && pendingCount > 0) {
            progress = false;
            for (int slot = 0; slot < pendingCount; slot++) {
                int f = pending[slot];
                if (!ge5(stats, fReqPk[f])) {
                    continue;
                }
                stats += fBonPk[f] - BIAS_5;
                if (fScore[f] > 0) {
                    posScoreStack[depth] -= fScore[f];
                    int[] b = bonRef[forcedItem[f]];
                    for (int s = 0; s < S; s++) {
                        if (b[s] > 0) {
                            remPosStack[d + s] -= b[s];
                        }
                    }
                }
                equippedForced[f] = true;
                closureLog[closureSize++] = f;
                // swap-remove from pending
                int last = pendingCount - 1;
                int moved = pending[last];
                pending[slot] = moved;
                pendingPos[moved] = slot;
                pending[last] = f;
                pendingPos[f] = last;
                pendingCount = last;
                added++;
                progress = true;
                slot--;
            }
        }
        statsPkStack[depth] = stats;
        return added;
    }

    /**
     * Undo closure additions past the saved size, in exact reverse LIFO order.
     * Each swap-remove parked the removed index at the slot that becomes
     * pendingCount again on undo, so membership restores by re-growing.
     */
    private void undoClosure(int savedSize) {
        while (closureSize > savedSize) {
            int f = closureLog[--closureSize];
            equippedForced[f] = false;
            pending[pendingCount] = f;
            pendingPos[f] = pendingCount;
            pendingCount++;
        }
    }

    private void markClosure() {
        for (int k = 0; k < closureSize; k++) {
            valid[forcedItem[closureLog[k]]] = true;
        }
    }

    private int pathWeight(long branchMask) {
        int weight = extractedScore;
        for (long m = branchMask; m != 0; m &= m - 1) {
            weight += bScore[Long.numberOfTrailingZeros(m)];
        }
        for (int k = 0; k < closureSize; k++) {
            weight += fScore[closureLog[k]];
        }
        return weight;
    }

    private int closureScoreFrom(int start) {
        int weight = 0;
        for (int k = start; k < closureSize; k++) {
            weight += fScore[closureLog[k]];
        }
        return weight;
    }

    private void initRemaining() {
        for (int s = 0; s < S; s++) {
            remPosStack[s] = 0;
        }
        int posScore = 0;
        for (int f = 0; f < forcedCount; f++) {
            int[] b = bonRef[forcedItem[f]];
            for (int s = 0; s < S; s++) {
                if (b[s] > 0) {
                    remPosStack[s] += b[s];
                }
            }
            if (fScore[f] > 0) {
                posScore += fScore[f];
            }
        }
        for (int p = 0; p < branchCount; p++) {
            int[] b = bonRef[branchItem[p]];
            for (int s = 0; s < S; s++) {
                if (b[s] > 0) {
                    remPosStack[s] += b[s];
                }
            }
            if (bScore[p] > 0) {
                posScore += bScore[p];
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
        for (int slot = 0; slot < pendingCount; slot++) {
            if (ge5(reach, fReqPk[pending[slot]])) {
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
            int[] b = bonRef[branchItem[p]];
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
            undoClosure(savedClosure);
        }
    }

    private void reconstruct(long branchMask) {
        Arrays.fill(equippedForced, 0, forcedCount, false);
        resetPending();
        long stats = extractedStatsPk;
        for (long m = branchMask; m != 0; m &= m - 1) {
            int p = Long.numberOfTrailingZeros(m);
            valid[branchItem[p]] = true;
            stats += bBonPk[p] - BIAS_5;
        }
        statsPkStack[0] = stats;
        closureSize = 0;
        runClosure(0);
        markClosure();
    }

}
