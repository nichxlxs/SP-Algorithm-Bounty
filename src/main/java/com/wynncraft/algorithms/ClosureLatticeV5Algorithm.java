package com.wynncraft.algorithms;

import com.wynncraft.core.WynnPlayer;
import com.wynncraft.core.interfaces.IAlgorithm;
import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.Information;
import com.wynncraft.enums.SkillPoint;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Faithful adoption of the Fred V2 submission's implementation idiom
 * (credit to Frederik): per-item five-field vector objects (direct field
 * loads in the hot loops), the worst-case-sufficiency greedy, compaction,
 * and the iterative mask search - with this series' robustness grafts as
 * the only changes:
 *
 *  - fixed 64-item working arrays (final: bounds checks fold) with scalar
 *    delegation above, instead of a hard 32-item bound;
 *  - a node budget bounding worst-case latency, best-found still returned;
 *  - fresh visited tables only while they are provably small (m <= 16,
 *    64 KB); a budget-bounded hash set beyond, and delegation to the scalar
 *    Closure Lattice V1 solver above 24 undetermined items - no input can
 *    crash the entry or exhaust memory (Fred V2 allocates boolean[2^N] per
 *    call, 32 MB at N=25).
 *
 * Semantics are the verified cascade rules; the fuzzer asserts equality
 * with V1-V4 on every oracle instance. No state carries between calls; no
 * equipment or player caching.
 */
@Information(name = "Closure Lattice", version = 5, authors = {"claude", "Frederik"})
public class ClosureLatticeV5Algorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();
    private static final int S = 5;
    private static final int UNDETERMINED_LIMIT = 24;
    /** Fixed capacity (final arrays let the JIT fold bounds checks); larger
     *  inputs delegate to the scalar solver, which has no cap. */
    private static final int MAX_ITEMS = 64;
    private static final int FRESH_TABLE_LIMIT = 16;
    private static final int NODE_BUDGET = 1 << 16;

    /** Fallback exact solver for oversized undetermined sets. */
    private final ClosureLatticeAlgorithm scalar = new ClosureLatticeAlgorithm();

    /** Five skill lanes as plain fields: direct loads in the hot loops. */
    private static final class Vec5 {
        int s0;
        int s1;
        int s2;
        int s3;
        int s4;

        void setFrom(int[] arr) {
            s0 = arr[0];
            s1 = arr[1];
            s2 = arr[2];
            s3 = arr[3];
            s4 = arr[4];
        }

        void setFrom(Vec5 o) {
            s0 = o.s0;
            s1 = o.s1;
            s2 = o.s2;
            s3 = o.s3;
            s4 = o.s4;
        }
    }

    private final Vec5[] itemReq = newVecs(MAX_ITEMS);
    private final Vec5[] itemBon = newVecs(MAX_ITEMS);
    private final boolean[] negItem = new boolean[MAX_ITEMS];
    private final boolean[] result = new boolean[MAX_ITEMS];
    private final int[] itemIdx = new int[MAX_ITEMS];
    private int[] stack = new int[1 << 16];

    private static Vec5[] newVecs(int n) {
        Vec5[] v = new Vec5[n];
        for (int i = 0; i < n; i++) {
            v[i] = new Vec5();
        }
        return v;
    }
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
        if (count > MAX_ITEMS) {
            return scalar.run(player);
        }

        int base0 = player.allocated(SKILL_POINTS[0]);
        int base1 = player.allocated(SKILL_POINTS[1]);
        int base2 = player.allocated(SKILL_POINTS[2]);
        int base3 = player.allocated(SKILL_POINTS[3]);
        int base4 = player.allocated(SKILL_POINTS[4]);
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

        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            if ((b[0] | b[1] | b[2] | b[3] | b[4]) == 0 && (r[0] | r[1] | r[2] | r[3] | r[4]) == 0) {
                result[i] = true;   // zero-stat items (tomes)
                negItem[i] = false;
                continue;
            }
            result[i] = false;
            itemReq[i].setFrom(r);
            Vec5 bv = itemBon[i];
            bv.setFrom(b);
            boolean neg = bv.s0 < 0 || bv.s1 < 0 || bv.s2 < 0 || bv.s3 < 0 || bv.s4 < 0;
            negItem[i] = neg;
            if (neg) {
                min0 += Math.min(bv.s0, 0);
                min1 += Math.min(bv.s1, 0);
                min2 += Math.min(bv.s2, 0);
                min3 += Math.min(bv.s3, 0);
                min4 += Math.min(bv.s4, 0);
            }
        }

        // Greedy over the worst-case bound; fixpoint.
        boolean added = true;
        while (added) {
            added = false;
            for (int i = 0; i < count; i++) {
                if (result[i] || negItem[i]) {
                    continue;
                }
                Vec5 r = itemReq[i];
                if ((r.s0 > 0 && cur0 + min0 < r.s0) || (r.s1 > 0 && cur1 + min1 < r.s1)
                    || (r.s2 > 0 && cur2 + min2 < r.s2) || (r.s3 > 0 && cur3 + min3 < r.s3)
                    || (r.s4 > 0 && cur4 + min4 < r.s4)) {
                    continue;
                }
                Vec5 b = itemBon[i];
                cur0 += b.s0;
                cur1 += b.s1;
                cur2 += b.s2;
                cur3 += b.s3;
                cur4 += b.s4;
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
                itemReq[m].setFrom(itemReq[i]);
                itemBon[m].setFrom(itemBon[i]);
                negItem[m] = negItem[i];
            }
            m++;
        }

        if (m > UNDETERMINED_LIMIT) {
            return scalar.run(player);
        }

        if (m > 0) {
            int bestCombo = search(m, cur0, cur1, cur2, cur3, cur4);
            for (int slot = 0; slot < m; slot++) {
                if ((bestCombo & (1 << slot)) != 0) {
                    result[itemIdx[slot]] = true;
                    Vec5 b = itemBon[slot];
                    cur0 += b.s0;
                    cur1 += b.s1;
                    cur2 += b.s2;
                    cur3 += b.s3;
                    cur4 += b.s4;
                }
            }
        }

        // The greedy/search stats already hold base + every valid bonus; the
        // modify vector is their difference - no item is ever re-read.
        List<IEquipment> validList = new ArrayList<>(count);
        List<IEquipment> invalidList = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (result[i]) {
                validList.add(item);
            } else {
                invalidList.add(item);
            }
        }
        if (!validList.isEmpty()) {
            bonusTotal[0] = cur0 - base0;
            bonusTotal[1] = cur1 - base1;
            bonusTotal[2] = cur2 - base2;
            bonusTotal[3] = cur3 - base3;
            bonusTotal[4] = cur4 - base4;
            player.modify(bonusTotal, true);
        }
        return new Result(validList, invalidList);
    }

    private int search(int m, int base0, int base1, int base2, int base3, int base4) {
        int full = (1 << m) - 1;
        boolean[] seen;
        if (m <= FRESH_TABLE_LIMIT) {
            seen = new boolean[1 << m];
            visitedLarge = null;
        } else {
            seen = null;
            visitedLarge = new HashSet<>();
        }
        int stackBound = (int) Math.min((long) (1 << m), (long) NODE_BUDGET * m + m + 1L);
        if (stack.length < stackBound) {
            stack = new int[stackBound];
        }

        int stackTop = 0;
        stack[stackTop++] = 0;
        if (seen != null) {
            seen[0] = true;
        } else {
            visitedLarge.add(0);
        }

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

            int c0 = base0;
            int c1 = base1;
            int c2 = base2;
            int c3 = base3;
            int c4 = base4;
            for (int rem = combo; rem != 0; rem &= rem - 1) {
                Vec5 b = itemBon[Integer.numberOfTrailingZeros(rem)];
                c0 += b.s0;
                c1 += b.s1;
                c2 += b.s2;
                c3 += b.s3;
                c4 += b.s4;
            }

            for (int slot = 0; slot < m; slot++) {
                int bit = 1 << slot;
                if ((combo & bit) != 0) {
                    continue;
                }
                int newCombo = combo | bit;
                if (seen != null) {
                    if (seen[newCombo]) {
                        continue;
                    }
                    seen[newCombo] = true;
                } else {
                    if (!visitedLarge.add(newCombo)) {
                        continue;
                    }
                }

                Vec5 r = itemReq[slot];
                if ((r.s0 > 0 && c0 < r.s0) || (r.s1 > 0 && c1 < r.s1)
                    || (r.s2 > 0 && c2 < r.s2) || (r.s3 > 0 && c3 < r.s3)
                    || (r.s4 > 0 && c4 < r.s4)) {
                    continue;
                }

                Vec5 ab = itemBon[slot];
                int n0 = c0 + ab.s0;
                int n1 = c1 + ab.s1;
                int n2 = c2 + ab.s2;
                int n3 = c3 + ab.s3;
                int n4 = c4 + ab.s4;

                if (negItem[slot]) {
                    boolean ok = true;
                    for (int rem = combo; rem != 0; rem &= rem - 1) {
                        int s2i = Integer.numberOfTrailingZeros(rem);
                        Vec5 r2 = itemReq[s2i];
                        Vec5 b2 = itemBon[s2i];
                        if ((r2.s0 > 0 && n0 - b2.s0 < r2.s0)
                            || (r2.s1 > 0 && n1 - b2.s1 < r2.s1)
                            || (r2.s2 > 0 && n2 - b2.s2 < r2.s2)
                            || (r2.s3 > 0 && n3 - b2.s3 < r2.s3)
                            || (r2.s4 > 0 && n4 - b2.s4 < r2.s4)) {
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


}
