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
 * Exact solver built on two structural facts of the problem:
 *
 * 1. The stat state after equipping a set of items is order-independent
 *    (base + sum of bonuses), so feasibility is a property of the subset
 *    lattice, not of permutations. The search space is masks, never orders.
 *
 * 2. Items with no negative bonus whose requirements only touch skills that
 *    no item in the input reduces ("forced" items) can be equipped greedily
 *    whenever their requirements are met, without ever costing optimality:
 *    their bonuses can only help others, the skills they depend on are
 *    monotonically non-decreasing over any equip sequence, and once equipped
 *    they can never be invalidated. Only the remaining "branch" items
 *    (negative bonuses, or requirements on a reducible skill) need search.
 *
 * Pipeline per call (no state is carried between calls; scratch arrays are
 * reused for memory only and every read cell is rewritten each run):
 *
 *   A. One cheap pass: cache requirement/bonus array references and collect
 *      the reducible-skill mask from items with negative bonuses (a
 *      precomputed flag on the equipment object).
 *   B. If no item has a negative bonus, the whole input is a monotone
 *      closure: equip anything equippable until a fixpoint - provably exact,
 *      no classification, no search. This is the common in-game shape.
 *   C. Otherwise classify into forced / branch, then run a greedy
 *      constructive attempt (forced closure interleaved with
 *      invariant-checked branch adds). If it equips everything, that is
 *      provably optimal and we stop.
 *   D. Exact DFS over branch-item masks with a visited set. Each node
 *      applies the forced closure fixpoint; transitions check the insertion
 *      rule (stats meet requirements) and the cascade invariant (stats stay
 *      at or above req+bonus of every equipped branch item). Best
 *      (count, weight) over all reachable masks is exact.
 *
 * The branch set is not artificially capped: masks are longs, the visited
 * set switches from a bitset to a hash set past 2^20 masks. Stat-identical
 * items are explored in canonical order (duplicates collapse to one subtree
 * per multiplicity), a permanently-blocked-item bound prunes hopeless
 * subtrees, and a node budget caps worst-case latency and memory on
 * adversarial inputs - if the budget is ever exhausted (requires tens of
 * pathological interacting negative items, far outside game data), the best
 * feasible set found so far is returned. Above 62 branch items, where exact
 * subset search is information-theoretically out of reach for any solver,
 * the result is the greedy constructive answer improved by leave-one-out
 * repair (re-running the greedy with each equipped negative item excluded).
 *
 * Supported numeric domain: per-lane requirements/bonuses and allocated SP
 * with magnitudes up to ~2^20 and item counts up to ~2^10. Real game data
 * stays below ~200; outside the stated domain int arithmetic could wrap.
 */
@Information(name = "Closure Lattice", version = 1, authors = {"claude"})
public class ClosureLatticeAlgorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();
    private static final int S = 5;
    /** Largest branch count solved with the flat visited bitset (2^20 bits = 128 KB). */
    private static final int VISITED_BITSET_LIMIT = 20;
    /** Largest branch count solved by exact subset search (long masks). */
    private static final int EXACT_LIMIT = 62;
    /**
     * Hard cap on DFS nodes per call. Real builds need well under a hundred
     * nodes; every repo test needs at most a few hundred. The cap only binds
     * on adversarial inputs with dozens of interacting negative items, where
     * it bounds latency and visited-set memory while still returning the best
     * feasible set found.
     */
    private static final int NODE_BUDGET = 1 << 16;
    /** Max leave-one-out repair attempts in the beyond-exact fallback. */
    private static final int REPAIR_ATTEMPTS = 64;

    // Per-item array references (read-only views of the equipment data).
    private int[][] reqRef = new int[0][];
    private int[][] bonRef = new int[0][];
    private boolean[] hasNeg = new boolean[0];
    private int[] itemScore = new int[0];        // filled only on DFS/fallback paths
    private boolean[] valid = new boolean[0];
    private final int[] base = new int[S];
    private final int[] bonusTotal = new int[S];

    // Classification (built only when negative bonuses exist).
    private int[] forcedIdx = new int[0];        // item indices, sorted by requirement sum
    private int[] forcedKeys = new int[0];
    private int[] branchIdx = new int[0];        // branch position -> item index
    private int[] branchKeys = new int[0];
    // Branch data, position-indexed flat [position * 5 + skill].
    private int[] bReq = new int[0];
    private int[] bBon = new int[0];
    private int[] bRpb = new int[0];             // req + bonus where req > 0, else MIN_VALUE
    private int[] bScore = new int[0];
    private long[] dupPred = new long[0];        // earlier positions with identical stats

    // Search scratch.
    private boolean[] equippedItem = new boolean[0];
    private int[] closureOrder = new int[0];
    private int[] statsStack = new int[0];       // depth * 5
    private int[] needStack = new int[0];
    private int[] remPosStack = new int[0];      // per-depth positive-bonus pool of unequipped items
    private int[] posScoreStack = new int[0];    // per-depth positive item-score pool
    private long[] visited = new long[0];
    private HashSet<Long> visitedLarge;

    // Per-run values.
    private int n;
    private int forcedCount;
    private int branchCount;
    private int closureSize;
    private int bestCount;
    private int bestWeight;
    private long bestMask;
    private boolean solved;
    private int nodesLeft;
    private int bannedItem;

    @Override
    public Result run(WynnPlayer player) {
        List<IEquipment> equipment = player.equipment();
        int count = equipment.size();
        List<IEquipment> validList = new ArrayList<>(count);
        List<IEquipment> invalidList = new ArrayList<>(count);
        if (count == 0) {
            return new Result(validList, invalidList);
        }

        this.n = count;
        ensureCapacity(count);
        for (int s = 0; s < S; s++) {
            base[s] = player.allocated(SKILL_POINTS[s]);
        }

        // Pass A: array refs + reducible-skill mask. hasNegativeBonus() is a
        // precomputed flag on the equipment, so items without negatives cost
        // two virtual calls and one boolean here - no array scan.
        int riskyMask = 0;
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            reqRef[i] = item.requirements();
            bonRef[i] = item.bonuses();
            boolean neg = item.hasNegativeBonus();
            hasNeg[i] = neg;
            if (neg) {
                int[] b = bonRef[i];
                for (int s = 0; s < S; s++) {
                    if (b[s] < 0) {
                        riskyMask |= 1 << s;
                    }
                }
            }
        }

        if (riskyMask == 0) {
            pureClosure();
        } else {
            solve(riskyMask);
        }

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
        reqRef = new int[cap][];
        bonRef = new int[cap][];
        hasNeg = new boolean[cap];
        itemScore = new int[cap];
        valid = new boolean[cap];
        forcedIdx = new int[cap];
        forcedKeys = new int[cap];
        branchIdx = new int[cap];
        branchKeys = new int[cap];
        bReq = new int[cap * S];
        bBon = new int[cap * S];
        bRpb = new int[cap * S];
        bScore = new int[cap];
        dupPred = new long[cap];
        equippedItem = new boolean[cap];
        closureOrder = new int[cap];
        statsStack = new int[(cap + 2) * S];
        needStack = new int[(cap + 2) * S];
        remPosStack = new int[(cap + 2) * S];
        posScoreStack = new int[cap + 2];
    }

    /**
     * No item reduces any skill, so feasibility is a monotone closure: equip
     * anything equippable until nothing changes. Exact by the standard
     * fixpoint argument; nothing equipped can ever be invalidated.
     */
    private void pureClosure() {
        Arrays.fill(valid, 0, n, false);
        int s0 = base[0];
        int s1 = base[1];
        int s2 = base[2];
        int s3 = base[3];
        int s4 = base[4];
        boolean progress = true;
        while (progress) {
            progress = false;
            for (int i = 0; i < n; i++) {
                if (valid[i]) {
                    continue;
                }
                int[] r = reqRef[i];
                if ((r[0] > 0 && s0 < r[0]) || (r[1] > 0 && s1 < r[1])
                    || (r[2] > 0 && s2 < r[2]) || (r[3] > 0 && s3 < r[3])
                    || (r[4] > 0 && s4 < r[4])) {
                    continue;
                }
                int[] b = bonRef[i];
                s0 += b[0];
                s1 += b[1];
                s2 += b[2];
                s3 += b[3];
                s4 += b[4];
                valid[i] = true;
                progress = true;
            }
        }
    }

    private void solve(int riskyMask) {
        classify(riskyMask);

        // --- Greedy constructive attempt at depth 0. ---
        // Note: remPosStack/posScoreStack are blindly decremented on this path
        // but never read; initRemaining() re-seeds them before the DFS reads.
        Arrays.fill(equippedItem, 0, n, false);
        bannedItem = -1;
        int greedyCount = runGreedy();
        long greedyMask = greedyBranchMask();

        if (greedyCount == n) {
            Arrays.fill(valid, 0, n, true);
            return;
        }

        if (branchCount == 0) {
            // Monotone closure is exact when nothing can be invalidated.
            System.arraycopy(equippedItem, 0, valid, 0, n);
            return;
        }

        fillItemScores();

        if (branchCount > EXACT_LIMIT) {
            fallbackWithRepair(greedyCount);
            return;
        }

        bestCount = greedyCount;
        bestWeight = pathWeight(greedyMask);
        bestMask = greedyMask;
        solved = false;
        nodesLeft = NODE_BUDGET;

        // --- Exact DFS over branch masks. ---
        prepareVisited();
        Arrays.fill(equippedItem, 0, n, false);
        initRemaining();
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
            needStack[s] = Integer.MIN_VALUE;
        }
        closureSize = 0;
        int rootCount = runClosure(0);
        int rootWeight = closureScoreFrom(0);
        dfs(0L, 0, rootCount, rootWeight);
        visitedLarge = null;

        reconstruct(bestMask);
    }

    private void classify(int riskyMask) {
        forcedCount = 0;
        branchCount = 0;
        for (int i = 0; i < n; i++) {
            int[] r = reqRef[i];
            int reqLanes = 0;
            int reqSum = 0;
            for (int s = 0; s < S; s++) {
                int rv = r[s];
                if (rv > 0) {
                    reqLanes |= 1 << s;
                    reqSum += rv;
                }
            }
            if (!hasNeg[i] && (reqLanes & riskyMask) == 0) {
                forcedIdx[forcedCount] = i;
                forcedKeys[forcedCount] = clampKey(reqSum);
                forcedCount++;
            } else {
                int[] b = bonRef[i];
                int p = branchCount++;
                branchIdx[p] = i;
                int off = p * S;
                int rpbSum = 0;
                int sc = 0;
                for (int s = 0; s < S; s++) {
                    int rv = r[s];
                    int bv = b[s];
                    bReq[off + s] = rv;
                    bBon[off + s] = bv;
                    bRpb[off + s] = rv > 0 ? rv + bv : Integer.MIN_VALUE;
                    if (rv > 0) {
                        rpbSum += rv + bv;
                    }
                    sc += bv;
                }
                bScore[p] = sc;
                // Positive branch items first (requirement sum ascending), then
                // negative items by req+bonus sum descending - the exchange-argument
                // order that succeeds on almost every feasible build.
                branchKeys[p] = hasNeg[i] ? (1 << 30) - clampKey(rpbSum) : clampKey(reqSum);
            }
        }
        insertionSort(forcedIdx, forcedKeys, forcedCount);
        sortBranch();

        // Canonical ordering for stat-identical branch items: position p may only
        // be equipped once every identical earlier position is equipped. Identical
        // items are interchangeable, so this collapses symmetric subtrees without
        // losing any (count, weight) outcome.
        for (int p = 0; p < branchCount && p < 64; p++) {
            long mask = 0L;
            int po = p * S;
            for (int q = 0; q < p; q++) {
                int qo = q * S;
                boolean same = true;
                for (int s = 0; s < S && same; s++) {
                    same = bReq[po + s] == bReq[qo + s] && bBon[po + s] == bBon[qo + s];
                }
                if (same) {
                    mask |= 1L << q;
                }
            }
            dupPred[p] = mask;
        }
    }

    /** Sorts branch positions by key, carrying the flat per-position data along. */
    private void sortBranch() {
        for (int i = 1; i < branchCount; i++) {
            int key = branchKeys[i];
            int idx = branchIdx[i];
            if (branchKeys[i - 1] < key || (branchKeys[i - 1] == key && branchIdx[i - 1] < idx)) {
                continue;
            }
            int sc = bScore[i];
            System.arraycopy(bReq, i * S, tmpLane, 0, S);
            System.arraycopy(bBon, i * S, tmpLane, S, S);
            System.arraycopy(bRpb, i * S, tmpLane, 2 * S, S);
            int j = i - 1;
            while (j >= 0 && (branchKeys[j] > key || (branchKeys[j] == key && branchIdx[j] > idx))) {
                branchKeys[j + 1] = branchKeys[j];
                branchIdx[j + 1] = branchIdx[j];
                bScore[j + 1] = bScore[j];
                System.arraycopy(bReq, j * S, bReq, (j + 1) * S, S);
                System.arraycopy(bBon, j * S, bBon, (j + 1) * S, S);
                System.arraycopy(bRpb, j * S, bRpb, (j + 1) * S, S);
                j--;
            }
            branchKeys[j + 1] = key;
            branchIdx[j + 1] = idx;
            bScore[j + 1] = sc;
            System.arraycopy(tmpLane, 0, bReq, (j + 1) * S, S);
            System.arraycopy(tmpLane, S, bBon, (j + 1) * S, S);
            System.arraycopy(tmpLane, 2 * S, bRpb, (j + 1) * S, S);
        }
    }

    private final int[] tmpLane = new int[3 * S];

    /**
     * Greedy constructive attempt: forced closure interleaved with
     * invariant-checked branch adds, at depth 0. Returns items equipped.
     * Honors {@link #bannedItem} (used by the beyond-exact repair).
     */
    private int runGreedy() {
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
            needStack[s] = Integer.MIN_VALUE;
        }
        closureSize = 0;
        int count = runClosure(0);
        boolean progress = branchCount > 0;
        while (progress) {
            progress = false;
            for (int p = 0; p < branchCount; p++) {
                int i = branchIdx[p];
                if (equippedItem[i] || i == bannedItem) {
                    continue;
                }
                int off = p * S;
                if (!branchReqMet(0, off) || !branchAddOk(0, off)) {
                    continue;
                }
                applyBranchInPlace(0, p);
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

    private void fillItemScores() {
        for (int i = 0; i < n; i++) {
            int[] b = bonRef[i];
            itemScore[i] = b[0] + b[1] + b[2] + b[3] + b[4];
        }
    }

    /**
     * Beyond {@link #EXACT_LIMIT} branch items exact subset search is out of
     * reach for any solver; return the greedy answer improved by leave-one-out
     * repair: one bad early pick (a big-bonus item whose negatives block many
     * others) is recovered by re-running the greedy with that item excluded.
     */
    private void fallbackWithRepair(int greedyCount) {
        System.arraycopy(equippedItem, 0, valid, 0, n);
        int bestC = greedyCount;
        int bestW = equippedScore();

        int attempts = 0;
        for (int p = 0; p < branchCount && attempts < REPAIR_ATTEMPTS; p++) {
            int i = branchIdx[p];
            if (!valid[i] || !hasNeg[i]) {
                continue;
            }
            attempts++;
            Arrays.fill(equippedItem, 0, n, false);
            bannedItem = i;
            int count = runGreedy();
            int weight = equippedScore();
            if (count > bestC || (count == bestC && weight > bestW)) {
                bestC = count;
                bestW = weight;
                System.arraycopy(equippedItem, 0, valid, 0, n);
            }
        }
        bannedItem = -1;
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

    private static int clampKey(int value) {
        if (value > (1 << 24)) {
            return 1 << 24;
        }
        if (value < -(1 << 24)) {
            return -(1 << 24);
        }
        return value;
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

    /** Resets the per-depth "still unequipped" positive-bonus aggregates at depth 0. */
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

    /** Check-and-mark; returns true when the mask was not seen before. */
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

    private boolean branchReqMet(int depth, int off) {
        int d = depth * S;
        return (bReq[off] <= 0 || statsStack[d] >= bReq[off])
            && (bReq[off + 1] <= 0 || statsStack[d + 1] >= bReq[off + 1])
            && (bReq[off + 2] <= 0 || statsStack[d + 2] >= bReq[off + 2])
            && (bReq[off + 3] <= 0 || statsStack[d + 3] >= bReq[off + 3])
            && (bReq[off + 4] <= 0 || statsStack[d + 4] >= bReq[off + 4]);
    }

    /** Cascade invariant after adding the item: new stats >= max(need, item rpb). */
    private boolean branchAddOk(int depth, int off) {
        int d = depth * S;
        for (int s = 0; s < S; s++) {
            int stat = statsStack[d + s] + bBon[off + s];
            if (stat < needStack[d + s] || stat < bRpb[off + s]) {
                return false;
            }
        }
        return true;
    }

    private void applyBranchInPlace(int depth, int p) {
        int off = p * S;
        int d = depth * S;
        for (int s = 0; s < S; s++) {
            int b = bBon[off + s];
            statsStack[d + s] += b;
            if (b > 0) {
                remPosStack[d + s] -= b;
            }
            int itemNeed = bRpb[off + s];
            if (itemNeed > needStack[d + s]) {
                needStack[d + s] = itemNeed;
            }
        }
        int sc = bScore[p];
        if (sc > 0) {
            posScoreStack[depth] -= sc;
        }
    }

    private void applyBranchToNext(int depth, int p) {
        int off = p * S;
        int d = depth * S;
        int nd = d + S;
        for (int s = 0; s < S; s++) {
            int b = bBon[off + s];
            statsStack[nd + s] = statsStack[d + s] + b;
            remPosStack[nd + s] = remPosStack[d + s] - (b > 0 ? b : 0);
            int need = needStack[d + s];
            int itemNeed = bRpb[off + s];
            needStack[nd + s] = itemNeed > need ? itemNeed : need;
        }
        int sc = bScore[p];
        posScoreStack[depth + 1] = posScoreStack[depth] - (sc > 0 ? sc : 0);
    }

    /**
     * Forced-closure fixpoint at the given depth: equips every forced item
     * whose requirements are met, updating stats in place. Forced items can
     * never violate the cascade invariant (their bonuses are non-negative and
     * their requirement skills are never reduced by any item), so no need
     * tracking is required for them. Returns the number of items equipped.
     */
    private int runClosure(int depth) {
        int d = depth * S;
        int added = 0;
        boolean progress = true;
        while (progress) {
            progress = false;
            for (int p = 0; p < forcedCount; p++) {
                int i = forcedIdx[p];
                if (equippedItem[i]) {
                    continue;
                }
                int[] r = reqRef[i];
                if ((r[0] > 0 && statsStack[d] < r[0])
                    || (r[1] > 0 && statsStack[d + 1] < r[1])
                    || (r[2] > 0 && statsStack[d + 2] < r[2])
                    || (r[3] > 0 && statsStack[d + 3] < r[3])
                    || (r[4] > 0 && statsStack[d + 4] < r[4])) {
                    continue;
                }
                int[] b = bonRef[i];
                for (int s = 0; s < S; s++) {
                    int bv = b[s];
                    statsStack[d + s] += bv;
                    if (bv > 0) {
                        remPosStack[d + s] -= bv;
                    }
                }
                int sc = b[0] + b[1] + b[2] + b[3] + b[4];
                if (sc > 0) {
                    posScoreStack[depth] -= sc;
                }
                equippedItem[i] = true;
                closureOrder[closureSize++] = i;
                added++;
                progress = true;
            }
        }
        return added;
    }

    /** Sum of scores of branch positions in the mask plus the current closure. */
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

        // Admissible bound: an unequipped item can only ever be added if each
        // required lane can still reach its requirement using every positive
        // bonus left in the pool. Items that cannot are excluded from the
        // achievable count; remaining weight is bounded by the positive scores
        // left. Prune when neither the count nor the weight tiebreak can beat
        // the incumbent.
        int d = depth * S;
        int countBound = count;
        for (int p = 0; p < branchCount; p++) {
            if ((mask & (1L << p)) != 0) {
                continue;
            }
            int off = p * S;
            boolean addable = true;
            for (int s = 0; s < S; s++) {
                int r = bReq[off + s];
                if (r > 0 && statsStack[d + s] + remPosStack[d + s] < r) {
                    addable = false;
                    break;
                }
            }
            if (addable) {
                countBound++;
            }
        }
        for (int p = 0; p < forcedCount; p++) {
            int i = forcedIdx[p];
            if (equippedItem[i]) {
                continue;
            }
            int[] r = reqRef[i];
            boolean addable = true;
            for (int s = 0; s < S; s++) {
                if (r[s] > 0 && statsStack[d + s] + remPosStack[d + s] < r[s]) {
                    addable = false;
                    break;
                }
            }
            if (addable) {
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
            int off = p * S;
            if (!branchReqMet(depth, off) || !branchAddOk(depth, off)) {
                continue;
            }
            long newMask = mask | bit;
            if (!visit(newMask)) {
                continue;
            }
            applyBranchToNext(depth, p);
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

    /**
     * Rebuild the valid set from the best branch mask. The forced closure is a
     * pure function of the branch set (forced requirements live on skills that
     * only ever increase), so recomputing the fixpoint at the final state
     * yields exactly the closure reached along the search path.
     */
    private void reconstruct(long branchMask) {
        Arrays.fill(valid, 0, n, false);
        Arrays.fill(equippedItem, 0, n, false);
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
        }
        for (long m = branchMask; m != 0; m &= m - 1) {
            int p = Long.numberOfTrailingZeros(m);
            int i = branchIdx[p];
            valid[i] = true;
            int off = p * S;
            for (int s = 0; s < S; s++) {
                statsStack[s] += bBon[off + s];
            }
        }
        closureSize = 0;
        runClosure(0);
        for (int k = 0; k < closureSize; k++) {
            valid[closureOrder[k]] = true;
        }
    }

}
