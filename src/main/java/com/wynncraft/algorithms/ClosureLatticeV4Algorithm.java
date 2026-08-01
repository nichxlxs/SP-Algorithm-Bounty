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
 * Hybrid entry: the minimal per-call shape proven fastest in the field
 * (preallocated flat arrays, worst-case-sufficiency greedy, compaction,
 * iterative mask search - the structure pioneered by the Fred V2 submission,
 * credit to Frederik) combined with this series' robustness guarantees:
 *
 *  - a reused visited bitset instead of a boolean[2^N] allocated per call
 *    (which costs 32 MB per invocation at 25 undetermined items);
 *  - a node budget bounding worst-case latency on adversarial inputs, with
 *    the best feasible combination found still returned;
 *  - no fixed item cap: beyond 24 undetermined items (outside any real or
 *    test input) the call delegates to the scalar Closure Lattice V1 solver,
 *    which remains exact to 62 branch items and degrades gracefully beyond.
 *
 * Search semantics are the verified cascade rules: an item can be added when
 * current totals meet its requirements, and after adding a negative item
 * every equipped item must still hold totals-minus-own-bonus >= requirement
 * (non-negative additions cannot break the invariant). The greedy phase
 * equips items whose requirements pass under minSP = current stats plus
 * every negative component in the input - a lower bound over all outcomes,
 * making those picks lexicographically dominant. No state carries between
 * calls; no equipment or player caching.
 */
@Information(name = "Closure Lattice", version = 4, authors = {"claude", "Frederik"})
public class ClosureLatticeV4Algorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();
    private static final int S = 5;
    /** Beyond this many undetermined items, delegate to the scalar solver. */
    private static final int UNDETERMINED_LIMIT = 24;
    private static final int VISITED_BITSET_LIMIT = 20;
    private static final int NODE_BUDGET = 1 << 16;

    /** Fallback exact solver for oversized undetermined sets. */
    private final ClosureLatticeAlgorithm scalar = new ClosureLatticeAlgorithm();

    // Per-item flat data [item * 5 + skill]; reused, rewritten every call.
    private int[] req = new int[0];
    private int[] bon = new int[0];
    private boolean[] negItem = new boolean[0];
    private boolean[] result = new boolean[0];
    private int[] itemIdx = new int[0];        // undetermined slot -> original index
    private int[] stack = new int[0];
    private long[] visited = new long[0];
    private HashSet<Integer> visitedLarge;
    private final int[] bonusTotal = new int[S];

    @Override
    public void clearCache() {
        scalar.clearCache();
    }

    @Override
    public Result run(WynnPlayer player) {
        List<IEquipment> equipment = player.equipment();
        int count = equipment.size();
        if (count == 0) {
            return new Result(new ArrayList<>(0), new ArrayList<>(0));
        }
        ensureCapacity(count);

        int base0 = player.allocated(SKILL_POINTS[0]);
        int base1 = player.allocated(SKILL_POINTS[1]);
        int base2 = player.allocated(SKILL_POINTS[2]);
        int base3 = player.allocated(SKILL_POINTS[3]);
        int base4 = player.allocated(SKILL_POINTS[4]);

        // Load + free filter + worst-case bound in one sweep.
        int cur0 = base0;
        int cur1 = base1;
        int cur2 = base2;
        int cur3 = base3;
        int cur4 = base4;
        int min0 = 0;
        int min1 = 0;
        int min2 = 0;
        int min3 = 0;
        int min4 = 0;
        for (int i = 0, off = 0; i < count; i++, off += S) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            boolean neg = item.hasNegativeBonus();
            negItem[i] = neg;
            if (!neg && (r[0] | r[1] | r[2] | r[3] | r[4]) == 0
                && (b[0] | b[1] | b[2] | b[3] | b[4]) == 0) {
                // Zero-stat item (tomes): unconditionally valid, never stored.
                result[i] = true;
                continue;
            }
            req[off] = r[0];
            req[off + 1] = r[1];
            req[off + 2] = r[2];
            req[off + 3] = r[3];
            req[off + 4] = r[4];
            bon[off] = b[0];
            bon[off + 1] = b[1];
            bon[off + 2] = b[2];
            bon[off + 3] = b[3];
            bon[off + 4] = b[4];
            result[i] = false;
            if (neg) {
                min0 += Math.min(b[0], 0);
                min1 += Math.min(b[1], 0);
                min2 += Math.min(b[2], 0);
                min3 += Math.min(b[3], 0);
                min4 += Math.min(b[4], 0);
            }
        }

        // Greedy: equip every non-negative item whose requirements pass under
        // the worst-case bound (current + all negative components); such an
        // item is equippable and never invalidated in ANY outcome. Fixpoint.
        boolean added = true;
        while (added) {
            added = false;
            for (int i = 0, off = 0; i < count; i++, off += S) {
                if (result[i] || negItem[i]) {
                    continue;
                }
                if ((req[off] > 0 && cur0 + min0 < req[off])
                    || (req[off + 1] > 0 && cur1 + min1 < req[off + 1])
                    || (req[off + 2] > 0 && cur2 + min2 < req[off + 2])
                    || (req[off + 3] > 0 && cur3 + min3 < req[off + 3])
                    || (req[off + 4] > 0 && cur4 + min4 < req[off + 4])) {
                    continue;
                }
                cur0 += bon[off];
                cur1 += bon[off + 1];
                cur2 += bon[off + 2];
                cur3 += bon[off + 3];
                cur4 += bon[off + 4];
                result[i] = true;
                added = true;
            }
        }

        // Compact the undetermined items.
        int m = 0;
        for (int i = 0; i < count; i++) {
            if (result[i]) {
                continue;
            }
            itemIdx[m] = i;
            if (m != i) {
                System.arraycopy(req, i * S, req, m * S, S);
                System.arraycopy(bon, i * S, bon, m * S, S);
            }
            negItem[m] = negItem[i];
            m++;
        }

        if (m > UNDETERMINED_LIMIT) {
            return scalar.run(player);
        }

        int bestCombo = 0;
        if (m == 1) {
            if ((req[0] <= 0 || cur0 >= req[0]) && (req[1] <= 0 || cur1 >= req[1])
                && (req[2] <= 0 || cur2 >= req[2]) && (req[3] <= 0 || cur3 >= req[3])
                && (req[4] <= 0 || cur4 >= req[4])) {
                bestCombo = 1;
            }
            if (bestCombo != 0) {
                result[itemIdx[0]] = true;
            }
        } else if (m == 2) {
            bestCombo = solve2(cur0, cur1, cur2, cur3, cur4);
            for (int slot = 0; slot < 2; slot++) {
                if ((bestCombo & (1 << slot)) != 0) {
                    result[itemIdx[slot]] = true;
                }
            }
        } else if (m > 0) {
            bestCombo = search(m, cur0, cur1, cur2, cur3, cur4);
            for (int slot = 0; slot < m; slot++) {
                if ((bestCombo & (1 << slot)) != 0) {
                    result[itemIdx[slot]] = true;
                }
            }
        }

        // Build result + one batched modify.
        List<IEquipment> validList = new ArrayList<>(count);
        List<IEquipment> invalidList = new ArrayList<>(count);
        for (int s = 0; s < S; s++) {
            bonusTotal[s] = 0;
        }
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (result[i]) {
                validList.add(item);
                int[] b = item.bonuses();
                bonusTotal[0] += b[0];
                bonusTotal[1] += b[1];
                bonusTotal[2] += b[2];
                bonusTotal[3] += b[3];
                bonusTotal[4] += b[4];
            } else {
                invalidList.add(item);
            }
        }
        if (!validList.isEmpty()) {
            player.modify(bonusTotal, true);
        }
        return new Result(validList, invalidList);
    }

    /**
     * Iterative exact search over undetermined-item masks. Combo stats are
     * recomputed per popped mask (order-independent), transitions check the
     * insertion rule, and negative additions re-verify the cascade invariant
     * for every combo member. Returns the best (count, bonus-total) mask.
     */
    private int search(int m, int base0, int base1, int base2, int base3, int base4) {
        int full = (1 << m) - 1;
        prepareVisited(m);
        int stackTop = 0;
        stack[stackTop++] = 0;
        markVisited(0);

        int bestCombo = 0;
        int bestCount = 0;
        int bestTotal = base0 + base1 + base2 + base3 + base4;
        int nodesLeft = NODE_BUDGET;

        while (stackTop > 0) {
            int combo = stack[--stackTop];
            if (combo == full) {
                break;
            }
            if (--nodesLeft < 0) {
                break;
            }

            // Recompute this combo's stats (pure function of the mask).
            int c0 = base0;
            int c1 = base1;
            int c2 = base2;
            int c3 = base3;
            int c4 = base4;
            for (int rem = combo; rem != 0; rem &= rem - 1) {
                int off = Integer.numberOfTrailingZeros(rem) * S;
                c0 += bon[off];
                c1 += bon[off + 1];
                c2 += bon[off + 2];
                c3 += bon[off + 3];
                c4 += bon[off + 4];
            }

            for (int slot = 0; slot < m; slot++) {
                int bit = 1 << slot;
                if ((combo & bit) != 0) {
                    continue;
                }
                int newCombo = combo | bit;
                if (isVisited(newCombo)) {
                    continue;
                }
                markVisited(newCombo);

                int off = slot * S;
                if ((req[off] > 0 && c0 < req[off])
                    || (req[off + 1] > 0 && c1 < req[off + 1])
                    || (req[off + 2] > 0 && c2 < req[off + 2])
                    || (req[off + 3] > 0 && c3 < req[off + 3])
                    || (req[off + 4] > 0 && c4 < req[off + 4])) {
                    continue;
                }

                int n0 = c0 + bon[off];
                int n1 = c1 + bon[off + 1];
                int n2 = c2 + bon[off + 2];
                int n3 = c3 + bon[off + 3];
                int n4 = c4 + bon[off + 4];

                if (negItem[slot]) {
                    // The added item reduces stats: re-verify the cascade
                    // invariant (totals minus own bonus >= requirements) for
                    // every previously added combo member.
                    boolean ok = true;
                    for (int rem = combo; rem != 0; rem &= rem - 1) {
                        int off2 = Integer.numberOfTrailingZeros(rem) * S;
                        if ((req[off2] > 0 && n0 - bon[off2] < req[off2])
                            || (req[off2 + 1] > 0 && n1 - bon[off2 + 1] < req[off2 + 1])
                            || (req[off2 + 2] > 0 && n2 - bon[off2 + 2] < req[off2 + 2])
                            || (req[off2 + 3] > 0 && n3 - bon[off2 + 3] < req[off2 + 3])
                            || (req[off2 + 4] > 0 && n4 - bon[off2 + 4] < req[off2 + 4])) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        continue;
                    }
                }

                int newCount = Integer.bitCount(newCombo);
                int newTotal = n0 + n1 + n2 + n3 + n4;
                if (newCount > bestCount || (newCount == bestCount && newTotal > bestTotal)) {
                    bestCombo = newCombo;
                    bestCount = newCount;
                    bestTotal = newTotal;
                }
                stack[stackTop++] = newCombo;
            }
        }
        visitedLarge = null;
        return bestCombo;
    }

    /**
     * Exact two-item case, fully unrolled. A pair is feasible via some order
     * iff the first inserts at current stats and, after both are applied,
     * each member holds totals-minus-own-bonus >= requirements.
     */
    private int solve2(int c0, int c1, int c2, int c3, int c4) {
        boolean v0 = meets(0, c0, c1, c2, c3, c4);
        boolean v1 = meets(S, c0, c1, c2, c3, c4);
        // Pair: both members' exclude-self totals must meet their requirements,
        // and at least one order must start legally.
        boolean pair = (v0 || v1)
            && meets(0, c0 + bon[S], c1 + bon[S + 1], c2 + bon[S + 2], c3 + bon[S + 3], c4 + bon[S + 4])
            && meets(S, c0 + bon[0], c1 + bon[1], c2 + bon[2], c3 + bon[3], c4 + bon[4]);
        if (pair) {
            // Verify a legal order exists: the startable item goes first.
            if (v0 || v1) {
                return 3;
            }
        }
        if (v0 && v1) {
            int t0 = bon[0] + bon[1] + bon[2] + bon[3] + bon[4];
            int t1 = bon[S] + bon[S + 1] + bon[S + 2] + bon[S + 3] + bon[S + 4];
            return t0 >= t1 ? 1 : 2;
        }
        if (v0) {
            return 1;
        }
        if (v1) {
            return 2;
        }
        return 0;
    }

    private boolean meets(int off, int c0, int c1, int c2, int c3, int c4) {
        return (req[off] <= 0 || c0 >= req[off])
            && (req[off + 1] <= 0 || c1 >= req[off + 1])
            && (req[off + 2] <= 0 || c2 >= req[off + 2])
            && (req[off + 3] <= 0 || c3 >= req[off + 3])
            && (req[off + 4] <= 0 || c4 >= req[off + 4]);
    }

    private void ensureCapacity(int items) {
        if (result.length >= items) {
            return;
        }
        int cap = Math.max(items, result.length * 2 + 8);
        req = new int[cap * S];
        bon = new int[cap * S];
        negItem = new boolean[cap];
        result = new boolean[cap];
        itemIdx = new int[cap];
    }

    private void prepareVisited(int m) {
        // Stack bound: each mask is pushed at most once, and the node budget
        // caps pops at NODE_BUDGET with at most m pushes per pop.
        int maxMasks = 1 << m;
        int stackBound = (int) Math.min((long) maxMasks, (long) NODE_BUDGET * m + m + 1L);
        if (stack.length < stackBound) {
            stack = new int[stackBound];
        }
        if (m <= VISITED_BITSET_LIMIT) {
            int words = Math.max(1, maxMasks >>> 6);
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

    private boolean isVisited(int mask) {
        if (visitedLarge != null) {
            return visitedLarge.contains(mask);
        }
        return (visited[mask >>> 6] & (1L << (mask & 63))) != 0;
    }

    private void markVisited(int mask) {
        if (visitedLarge != null) {
            visitedLarge.add(mask);
        } else {
            visited[mask >>> 6] |= 1L << (mask & 63);
        }
    }

}
