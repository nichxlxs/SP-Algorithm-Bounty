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
 *   A. Snapshot + classify items into zero / forced / branch.
 *   B. Greedy constructive attempt (forced closure interleaved with
 *      invariant-checked branch adds). If it equips everything, that is
 *      provably optimal and we stop - the common case for real builds.
 *   C. Otherwise, exact DFS over branch-item masks with a visited bitset.
 *      Each node applies the forced closure fixpoint; transitions check the
 *      insertion rule (stats meet requirements) and the cascade invariant
 *      (stats stay at or above req+bonus of every equipped branch item).
 *      Best (count, weight) over all reachable masks is exact.
 *
 * The branch set is not artificially capped: masks are longs, the visited
 * set switches from a bitset to a hash set past 2^20 masks, and above 62
 * branch items (far beyond any real or test input) the greedy result is
 * returned rather than crashing. Stat-identical items are explored in a
 * canonical order (duplicates collapse to one subtree per multiplicity), a
 * permanently-blocked-item bound prunes hopeless subtrees, and a node budget
 * caps worst-case latency and memory on adversarial inputs - if the budget
 * is ever exhausted (requires tens of pathological branch items, far outside
 * game data), the best feasible set found so far is returned.
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
    /** Largest branch count solved exactly at all (long masks). */
    private static final int EXACT_LIMIT = 62;
    /**
     * Hard cap on DFS nodes per call. Real builds need well under a hundred
     * nodes; every repo test needs at most a few hundred. The cap only binds
     * on adversarial inputs with dozens of interacting negative items, where
     * it bounds latency and visited-set memory while still returning the best
     * feasible set found.
     */
    private static final int NODE_BUDGET = 1 << 16;

    // Per-item snapshot, flat [item * 5 + skill].
    private int[] req = new int[0];
    private int[] bon = new int[0];
    private int[] rpb = new int[0];      // req + bonus where req > 0, else MIN_VALUE
    private int[] score = new int[0];    // sum of the item's 5 bonuses
    private boolean[] valid = new boolean[0];

    // Classification (item indices) + sort keys.
    private int[] zeroIdx = new int[0];
    private int[] forcedIdx = new int[0];
    private int[] branchIdx = new int[0];
    private int[] forcedKeys = new int[0];
    private int[] branchKeys = new int[0];

    // Search scratch.
    private boolean[] equippedItem = new boolean[0];
    private int[] closureOrder = new int[0];
    private int[] statsStack = new int[0];   // (depth) * 5
    private int[] needStack = new int[0];
    private int[] remPosStack = new int[0];  // per-depth: sum of positive bonuses of unequipped items, per lane
    private int[] posScoreStack = new int[0]; // per-depth: sum of positive item scores of unequipped items
    private long[] dupPred = new long[0];    // per branch position: earlier positions with identical stats
    private long[] visited = new long[0];
    private HashSet<Long> visitedLarge;

    // Per-run values.
    private int n;
    private int zeroCount;
    private int forcedCount;
    private int branchCount;
    private int closureSize;
    private final int[] base = new int[S];
    private int bestCount;
    private int bestWeight;
    private long bestMask;
    private boolean solved;
    private int nodesLeft;

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

        snapshot(equipment);
        solve();

        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (valid[i]) {
                validList.add(item);
                player.modify(item.bonuses(), true);
            } else {
                invalidList.add(item);
            }
        }
        return new Result(validList, invalidList);
    }

    private void ensureCapacity(int items) {
        if (score.length >= items) {
            return;
        }
        int cap = Math.max(items, score.length * 2 + 8);
        req = new int[cap * S];
        bon = new int[cap * S];
        rpb = new int[cap * S];
        score = new int[cap];
        valid = new boolean[cap];
        zeroIdx = new int[cap];
        forcedIdx = new int[cap];
        branchIdx = new int[cap];
        forcedKeys = new int[cap];
        branchKeys = new int[cap];
        equippedItem = new boolean[cap];
        closureOrder = new int[cap];
        statsStack = new int[(cap + 2) * S];
        needStack = new int[(cap + 2) * S];
        remPosStack = new int[(cap + 2) * S];
        posScoreStack = new int[cap + 2];
        dupPred = new long[cap];
    }

    private void snapshot(List<IEquipment> equipment) {
        int riskyMask = 0;
        for (int i = 0, off = 0; i < n; i++, off += S) {
            IEquipment item = equipment.get(i);
            int[] itemReq = item.requirements();
            int[] itemBon = item.bonuses();
            int itemScore = 0;
            for (int s = 0; s < S; s++) {
                int r = itemReq[s];
                int b = itemBon[s];
                req[off + s] = r;
                bon[off + s] = b;
                rpb[off + s] = r > 0 ? r + b : Integer.MIN_VALUE;
                itemScore += b;
                if (b < 0) {
                    riskyMask |= 1 << s;
                }
            }
            score[i] = itemScore;
        }

        zeroCount = 0;
        forcedCount = 0;
        branchCount = 0;
        for (int i = 0, off = 0; i < n; i++, off += S) {
            boolean hasReq = false;
            boolean hasNeg = false;
            boolean hasPos = false;
            boolean reqOnRisky = false;
            int reqSum = 0;
            int rpbSum = 0;
            for (int s = 0; s < S; s++) {
                int r = req[off + s];
                int b = bon[off + s];
                if (r > 0) {
                    hasReq = true;
                    reqSum += r;
                    rpbSum += r + b;
                    if ((riskyMask & (1 << s)) != 0) {
                        reqOnRisky = true;
                    }
                }
                if (b < 0) {
                    hasNeg = true;
                } else if (b > 0) {
                    hasPos = true;
                }
            }
            if (!hasReq && !hasNeg && !hasPos) {
                zeroIdx[zeroCount++] = i;
            } else if (!hasNeg && !reqOnRisky) {
                forcedIdx[forcedCount] = i;
                forcedKeys[forcedCount] = clampKey(reqSum);
                forcedCount++;
            } else {
                branchIdx[branchCount] = i;
                // Positive branch items first (requirement sum ascending), then
                // negative items by req+bonus sum descending - the exchange-argument
                // order that succeeds on almost every feasible build.
                branchKeys[branchCount] = hasNeg
                    ? (1 << 30) - clampKey(rpbSum)
                    : clampKey(reqSum);
                branchCount++;
            }
        }
        insertionSort(forcedIdx, forcedKeys, forcedCount);
        insertionSort(branchIdx, branchKeys, branchCount);

        // Canonical ordering for stat-identical branch items: position p may only
        // be equipped once every identical earlier position is equipped. Identical
        // items are interchangeable, so this collapses symmetric subtrees without
        // losing any (count, weight) outcome.
        for (int p = 0; p < branchCount && p < 64; p++) {
            long mask = 0L;
            int pi = branchIdx[p] * S;
            for (int q = 0; q < p; q++) {
                int qi = branchIdx[q] * S;
                boolean same = true;
                for (int s = 0; s < S && same; s++) {
                    same = req[pi + s] == req[qi + s] && bon[pi + s] == bon[qi + s];
                }
                if (same) {
                    mask |= 1L << q;
                }
            }
            dupPred[p] = mask;
        }
    }

    /** Resets the per-depth "still unequipped" positive-bonus aggregates at depth 0. */
    private void initRemaining() {
        for (int s = 0; s < S; s++) {
            remPosStack[s] = 0;
        }
        int posScore = 0;
        for (int i = 0, off = 0; i < n; i++, off += S) {
            for (int s = 0; s < S; s++) {
                int b = bon[off + s];
                if (b > 0) {
                    remPosStack[s] += b;
                }
            }
            if (score[i] > 0) {
                posScore += score[i];
            }
        }
        posScoreStack[0] = posScore;
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

    private void solve() {
        // --- Phase B: greedy constructive attempt at depth 0. ---
        Arrays.fill(equippedItem, 0, n, false);
        initRemaining();
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
            needStack[s] = Integer.MIN_VALUE;
        }
        for (int k = 0; k < zeroCount; k++) {
            equippedItem[zeroIdx[k]] = true;
        }
        closureSize = 0;
        int greedyCount = zeroCount + runClosure(0);
        long greedyMask = 0L;
        boolean progress = branchCount > 0;
        while (progress) {
            progress = false;
            for (int p = 0; p < branchCount; p++) {
                int i = branchIdx[p];
                if (equippedItem[i]) {
                    continue;
                }
                int ioff = i * S;
                if (!reqMet(0, ioff) || !branchAddOk(0, ioff)) {
                    continue;
                }
                applyBranchInPlace(0, ioff);
                equippedItem[i] = true;
                if (p < 64) {
                    greedyMask |= 1L << p;
                }
                greedyCount++;
                greedyCount += runClosure(0);
                progress = true;
            }
        }

        if (greedyCount == n) {
            Arrays.fill(valid, 0, n, true);
            return;
        }

        if (branchCount == 0) {
            // Monotone closure is exact when nothing can be invalidated.
            System.arraycopy(equippedItem, 0, valid, 0, n);
            return;
        }

        if (branchCount > EXACT_LIMIT) {
            // Beyond exact reach at any conceivable budget; return the greedy
            // result straight from the equip flags (mask arithmetic is unsafe
            // past 64 branch positions and is not used on this path).
            System.arraycopy(equippedItem, 0, valid, 0, n);
            return;
        }

        bestCount = greedyCount;
        bestWeight = pathWeight(greedyMask);
        bestMask = greedyMask;
        solved = false;
        nodesLeft = NODE_BUDGET;

        // --- Phase C: exact DFS over branch masks. ---
        prepareVisited();
        Arrays.fill(equippedItem, 0, n, false);
        initRemaining();
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
            needStack[s] = Integer.MIN_VALUE;
        }
        closureSize = 0;
        int rootCount = zeroCount + runClosure(0);
        int rootWeight = closureScoreFrom(0);
        dfs(0L, 0, rootCount, rootWeight);
        visitedLarge = null;

        reconstruct(bestMask);
    }

    /** Sum of scores of branch items in the mask plus the closure they enable. */
    private int pathWeight(long branchMask) {
        int weight = 0;
        for (long m = branchMask; m != 0; m &= m - 1) {
            weight += score[branchIdx[Long.numberOfTrailingZeros(m)]];
        }
        for (int k = 0; k < closureSize; k++) {
            weight += score[closureOrder[k]];
        }
        return weight;
    }

    private int closureScoreFrom(int start) {
        int weight = 0;
        for (int k = start; k < closureSize; k++) {
            weight += score[closureOrder[k]];
        }
        return weight;
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

    private boolean reqMet(int depth, int ioff) {
        int off = depth * S;
        return (req[ioff] <= 0 || statsStack[off] >= req[ioff])
            && (req[ioff + 1] <= 0 || statsStack[off + 1] >= req[ioff + 1])
            && (req[ioff + 2] <= 0 || statsStack[off + 2] >= req[ioff + 2])
            && (req[ioff + 3] <= 0 || statsStack[off + 3] >= req[ioff + 3])
            && (req[ioff + 4] <= 0 || statsStack[off + 4] >= req[ioff + 4]);
    }

    /** Cascade invariant after adding the item: new stats >= max(need, item rpb). */
    private boolean branchAddOk(int depth, int ioff) {
        int off = depth * S;
        for (int s = 0; s < S; s++) {
            int stat = statsStack[off + s] + bon[ioff + s];
            int need = needStack[off + s];
            int itemNeed = rpb[ioff + s];
            if (stat < need || stat < itemNeed) {
                return false;
            }
        }
        return true;
    }

    private void applyBranchInPlace(int depth, int ioff) {
        int off = depth * S;
        for (int s = 0; s < S; s++) {
            int b = bon[ioff + s];
            statsStack[off + s] += b;
            if (b > 0) {
                remPosStack[off + s] -= b;
            }
            int itemNeed = rpb[ioff + s];
            if (itemNeed > needStack[off + s]) {
                needStack[off + s] = itemNeed;
            }
        }
        int itemScore = score[ioff / S];
        if (itemScore > 0) {
            posScoreStack[depth] -= itemScore;
        }
    }

    private void applyBranchToNext(int depth, int ioff) {
        int off = depth * S;
        int noff = off + S;
        for (int s = 0; s < S; s++) {
            int b = bon[ioff + s];
            statsStack[noff + s] = statsStack[off + s] + b;
            remPosStack[noff + s] = remPosStack[off + s] - (b > 0 ? b : 0);
            int need = needStack[off + s];
            int itemNeed = rpb[ioff + s];
            needStack[noff + s] = itemNeed > need ? itemNeed : need;
        }
        int itemScore = score[ioff / S];
        posScoreStack[depth + 1] = posScoreStack[depth] - (itemScore > 0 ? itemScore : 0);
    }

    /**
     * Forced-closure fixpoint at the given depth: equips every forced item
     * whose requirements are met, updating stats in place. Forced items can
     * never violate the cascade invariant (their bonuses are non-negative and
     * their requirement skills are never reduced by any item), so no need
     * tracking is required for them. Returns the number of items equipped.
     */
    private int runClosure(int depth) {
        int off = depth * S;
        int added = 0;
        boolean progress = true;
        while (progress) {
            progress = false;
            for (int p = 0; p < forcedCount; p++) {
                int i = forcedIdx[p];
                if (equippedItem[i]) {
                    continue;
                }
                int ioff = i * S;
                if ((req[ioff] > 0 && statsStack[off] < req[ioff])
                    || (req[ioff + 1] > 0 && statsStack[off + 1] < req[ioff + 1])
                    || (req[ioff + 2] > 0 && statsStack[off + 2] < req[ioff + 2])
                    || (req[ioff + 3] > 0 && statsStack[off + 3] < req[ioff + 3])
                    || (req[ioff + 4] > 0 && statsStack[off + 4] < req[ioff + 4])) {
                    continue;
                }
                for (int s = 0; s < S; s++) {
                    int b = bon[ioff + s];
                    statsStack[off + s] += b;
                    if (b > 0) {
                        remPosStack[off + s] -= b;
                    }
                }
                if (score[i] > 0) {
                    posScoreStack[depth] -= score[i];
                }
                equippedItem[i] = true;
                closureOrder[closureSize++] = i;
                added++;
                progress = true;
            }
        }
        return added;
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
        int countBound = count;
        for (int p = 0; p < branchCount; p++) {
            if ((mask & (1L << p)) != 0) {
                continue;
            }
            if (everAddable(depth, branchIdx[p] * S)) {
                countBound++;
            }
        }
        for (int p = 0; p < forcedCount; p++) {
            int i = forcedIdx[p];
            if (!equippedItem[i] && everAddable(depth, i * S)) {
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
            int i = branchIdx[p];
            int ioff = i * S;
            if (!reqMet(depth, ioff) || !branchAddOk(depth, ioff)) {
                continue;
            }
            long newMask = mask | bit;
            if (!visit(newMask)) {
                continue;
            }
            applyBranchToNext(depth, ioff);
            int savedClosure = closureSize;
            int added = runClosure(depth + 1);
            int addedWeight = score[i] + closureScoreFrom(savedClosure);
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

    /** Can this item's requirements still be met in ANY extension from this node? */
    private boolean everAddable(int depth, int ioff) {
        int off = depth * S;
        for (int s = 0; s < S; s++) {
            int r = req[ioff + s];
            if (r > 0 && statsStack[off + s] + remPosStack[off + s] < r) {
                return false;
            }
        }
        return true;
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
        for (int k = 0; k < zeroCount; k++) {
            valid[zeroIdx[k]] = true;
        }
        for (int s = 0; s < S; s++) {
            statsStack[s] = base[s];
        }
        for (long m = branchMask; m != 0; m &= m - 1) {
            int i = branchIdx[Long.numberOfTrailingZeros(m)];
            valid[i] = true;
            int ioff = i * S;
            for (int s = 0; s < S; s++) {
                statsStack[s] += bon[ioff + s];
            }
        }
        closureSize = 0;
        runClosure(0);
        for (int k = 0; k < closureSize; k++) {
            valid[closureOrder[k]] = true;
        }
    }

}
