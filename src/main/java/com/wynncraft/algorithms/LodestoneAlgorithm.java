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
 * Lodestone works in three steps:
 *
 * 1. Stat-less items (tomes etc.) are valid by definition, skip them.
 * 2. If an item with no downsides passes its requirements even in the worst
 *    case (as if every negative bonus in the build were already applied),
 *    nothing can ever invalidate it - equip it right away. Repeat until
 *    nothing more unlocks. On most real builds this settles everything.
 * 3. Whatever is left (usually 0-3 items) goes through an exhaustive search
 *    over item combinations, checking the cascade rule on the way: after a
 *    negative item is added, every equipped item still has to meet its
 *    requirements without counting its own bonus.
 *
 * The search is bounded so no input can freeze or OOM the server: dedupe
 * tables stay small, the node count is capped, and inputs too big for the
 * bitmask search (way past anything the game can produce) fall back to the
 * slower uncapped solver instead of crashing.
 *
 * The step 2/3 structure follows Frederik's Fred Algo V2 - credit where due,
 * that shape benches fastest by a good margin.
 */
@Information(name = "Lodestone", version = 1, authors = {"nichxlxs", "Frederik"})
public class LodestoneAlgorithm implements IAlgorithm<WynnPlayer> {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();

    // Past these sizes we hand off to the fallback solver rather than risk
    // the fixed buffers or the bitmask width.
    private static final int MAX_ITEMS = 64;
    private static final int MAX_UNDETERMINED = 24;

    // A fresh dedupe table is cheap while 2^m fits in 64KB; past that use a
    // hash set so a weird input can't allocate megabytes per call.
    private static final int SMALL_TABLE_LIMIT = 16;
    private static final int MAX_NODES = 1 << 16;

    private final LodestoneFallback fallback = new LodestoneFallback();

    // Five stats as plain fields so the hot loops are direct field loads.
    private static final class Stats {
        int str;
        int dex;
        int intel;
        int def;
        int agi;

        void setFrom(int[] arr) {
            str = arr[0];
            dex = arr[1];
            intel = arr[2];
            def = arr[3];
            agi = arr[4];
        }

        // A requirement that isn't positive means nothing (confirmed by the
        // maintainer), so store a sentinel no stat can be below - the checks
        // then need no zero-guard.
        void setFromClamped(int[] arr) {
            str = arr[0] > 0 ? arr[0] : Integer.MIN_VALUE;
            dex = arr[1] > 0 ? arr[1] : Integer.MIN_VALUE;
            intel = arr[2] > 0 ? arr[2] : Integer.MIN_VALUE;
            def = arr[3] > 0 ? arr[3] : Integer.MIN_VALUE;
            agi = arr[4] > 0 ? arr[4] : Integer.MIN_VALUE;
        }

        void setFrom(Stats o) {
            str = o.str;
            dex = o.dex;
            intel = o.intel;
            def = o.def;
            agi = o.agi;
        }
    }

    // Working buffers, overwritten from scratch on every call. Fixed size on
    // purpose: final arrays of constant length keep the loops tight. Validity
    // lives in a bitmask (the fast path handles at most 64 items) so the hot
    // loops never store into an array for it.
    private final Stats[] itemReqs = makeStats(MAX_ITEMS);
    private final Stats[] itemBonuses = makeStats(MAX_ITEMS);
    private final boolean[] negItems = new boolean[MAX_ITEMS];
    private final int[] itemIdxs = new int[MAX_ITEMS];
    private final int[] pendingIdxs = new int[MAX_ITEMS];
    private final int[] bonusDelta = new int[5];
    private int[] comboStack = new int[1 << 16];
    private HashSet<Integer> seenLarge;

    private static Stats[] makeStats(int n) {
        Stats[] out = new Stats[n];
        for (int i = 0; i < n; i++) {
            out[i] = new Stats();
        }
        return out;
    }

    @Override
    public Result run(WynnPlayer player) {
        List<IEquipment> equipment = player.equipment();
        int count = equipment.size();
        if (count == 0) {
            return new Result(new ArrayList<>(0), new ArrayList<>(0));
        }
        if (count > MAX_ITEMS) {
            return fallback.run(player);
        }

        int baseStr = player.allocated(SKILL_POINTS[0]);
        int baseDex = player.allocated(SKILL_POINTS[1]);
        int baseInt = player.allocated(SKILL_POINTS[2]);
        int baseDef = player.allocated(SKILL_POINTS[3]);
        int baseAgi = player.allocated(SKILL_POINTS[4]);

        // worst* = stats with everything valid so far applied, assuming every
        // negative bonus in the build hits us too. If an item passes its reqs
        // against worst*, it is safe in every possible outcome. The greedy
        // phase only accepts items with no downsides, so the gap between
        // worst* and the real stats stays fixed at the negative sums - one
        // set of accumulators is enough, the real stats fall out at the end.
        int worstStr = baseStr;
        int worstDex = baseDex;
        int worstInt = baseInt;
        int worstDef = baseDef;
        int worstAgi = baseAgi;

        long validMask = 0L;
        long fullMask = count == 64 ? -1L : (1L << count) - 1;

        // First sweep: classify items and collect the negative bonus sums.
        // Only negative items need their stats copied here - everything else
        // is either done (stat-less) or gets read again straight off the item
        // in the greedy sweep, which stores stats only for items that fail.
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            if ((b[0] | b[1] | b[2] | b[3] | b[4]) == 0 && (r[0] | r[1] | r[2] | r[3] | r[4]) == 0) {
                // tomes and other stat-less items are always fine
                validMask |= 1L << i;
                negItems[i] = false;
                continue;
            }
            boolean neg = b[0] < 0 || b[1] < 0 || b[2] < 0 || b[3] < 0 || b[4] < 0;
            negItems[i] = neg;
            if (neg) {
                itemReqs[i].setFromClamped(r);
                itemBonuses[i].setFrom(b);
                worstStr += Math.min(b[0], 0);
                worstDex += Math.min(b[1], 0);
                worstInt += Math.min(b[2], 0);
                worstDef += Math.min(b[3], 0);
                worstAgi += Math.min(b[4], 0);
            }
        }

        int negStr = worstStr - baseStr;
        int negDex = worstDex - baseDex;
        int negInt = worstInt - baseInt;
        int negDef = worstDef - baseDef;
        int negAgi = worstAgi - baseAgi;

        // Step 2: keep equipping items that are safe under the worst case,
        // until a full pass adds nothing. First pass over everything; repeat
        // passes only look at what's still pending.
        int pending = 0;
        boolean added = false;
        for (int i = 0; i < count; i++) {
            if ((validMask >>> i & 1L) != 0 || negItems[i]) {
                continue;
            }
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            if ((r[0] > 0 && worstStr < r[0]) || (r[1] > 0 && worstDex < r[1])
                || (r[2] > 0 && worstInt < r[2]) || (r[3] > 0 && worstDef < r[3])
                || (r[4] > 0 && worstAgi < r[4])) {
                itemReqs[i].setFromClamped(r);
                itemBonuses[i].setFrom(item.bonuses());
                pendingIdxs[pending++] = i;
                continue;
            }
            int[] b = item.bonuses();
            worstStr += b[0];
            worstDex += b[1];
            worstInt += b[2];
            worstDef += b[3];
            worstAgi += b[4];
            validMask |= 1L << i;
            added = true;
        }
        while (added && pending > 0) {
            added = false;
            for (int k = 0; k < pending; k++) {
                int i = pendingIdxs[k];
                Stats r = itemReqs[i];
                if (worstStr < r.str || worstDex < r.dex || worstInt < r.intel
                    || worstDef < r.def || worstAgi < r.agi) {
                    continue;
                }
                Stats b = itemBonuses[i];
                worstStr += b.str;
                worstDex += b.dex;
                worstInt += b.intel;
                worstDef += b.def;
                worstAgi += b.agi;
                validMask |= 1L << i;
                pendingIdxs[k--] = pendingIdxs[--pending];
                added = true;
            }
        }

        int curStr = worstStr - negStr;
        int curDex = worstDex - negDex;
        int curInt = worstInt - negInt;
        int curDef = worstDef - negDef;
        int curAgi = worstAgi - negAgi;

        // Walk the undecided bits to the front of the buffers so the search
        // only ever looks at those slots. Every undecided item already has
        // its stats stored - pending items got them on failure, negative
        // items in the first sweep.
        int undecided = 0;
        for (long rem = ~validMask & fullMask; rem != 0; rem &= rem - 1) {
            int i = Long.numberOfTrailingZeros(rem);
            itemIdxs[undecided] = i;
            if (undecided != i) {
                itemReqs[undecided].setFrom(itemReqs[i]);
                itemBonuses[undecided].setFrom(itemBonuses[i]);
                negItems[undecided] = negItems[i];
            }
            undecided++;
        }

        if (undecided > MAX_UNDETERMINED) {
            return fallback.run(player);
        }

        if (undecided > 0) {
            int bestCombo = findBestCombo(undecided, curStr, curDex, curInt, curDef, curAgi);
            for (int slot = 0; slot < undecided; slot++) {
                if ((bestCombo & (1 << slot)) != 0) {
                    validMask |= 1L << itemIdxs[slot];
                    Stats b = itemBonuses[slot];
                    curStr += b.str;
                    curDex += b.dex;
                    curInt += b.intel;
                    curDef += b.def;
                    curAgi += b.agi;
                }
            }
        }

        int validCount = Long.bitCount(validMask);
        // cur* already equals base + every valid bonus, so the player update
        // is just the difference. No need to touch the items again.
        if (validCount > 0) {
            bonusDelta[0] = curStr - baseStr;
            bonusDelta[1] = curDex - baseDex;
            bonusDelta[2] = curInt - baseInt;
            bonusDelta[3] = curDef - baseDef;
            bonusDelta[4] = curAgi - baseAgi;
            player.modify(bonusDelta, true);
        }
        if (validMask == fullMask) {
            // Everything fits - the usual case on real builds. The player's
            // equipment list is exactly the valid list.
            return new Result(equipment, new ArrayList<>(0));
        }
        IEquipment[] validItems = new IEquipment[validCount];
        IEquipment[] invalidItems = new IEquipment[count - validCount];
        int v = 0;
        int inv = 0;
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if ((validMask >>> i & 1L) != 0) {
                validItems[v++] = item;
            } else {
                invalidItems[inv++] = item;
            }
        }
        return new Result(Arrays.asList(validItems), Arrays.asList(invalidItems));
    }

    /**
     * Exhaustive search over the undecided items. Combos are bitmasks; each
     * combo's stats only depend on which items are in it, not the order, so
     * every combo needs checking once. Returns the combo with the most items,
     * ties broken by highest stat total.
     */
    private int findBestCombo(int slots, int baseStr, int baseDex, int baseInt, int baseDef, int baseAgi) {
        int full = (1 << slots) - 1;
        boolean[] seen;
        if (slots <= SMALL_TABLE_LIMIT) {
            seen = new boolean[1 << slots];
            seenLarge = null;
        } else {
            seen = null;
            seenLarge = new HashSet<>();
        }
        // Each combo is pushed at most once, and the node cap limits pushes
        // too, so this never needs the full 2^slots in pathological cases.
        int stackNeeded = (int) Math.min((long) (1 << slots), (long) MAX_NODES * slots + slots + 1L);
        if (comboStack.length < stackNeeded) {
            comboStack = new int[stackNeeded];
        }

        int top = 0;
        comboStack[top++] = 0;
        if (seen != null) {
            seen[0] = true;
        } else {
            seenLarge.add(0);
        }

        int bestCombo = 0;
        int bestCount = 0;
        int bestTotal = baseStr + baseDex + baseInt + baseDef + baseAgi;
        int nodesLeft = MAX_NODES;

        while (top > 0) {
            int combo = comboStack[--top];
            if (combo == full) {
                break; // everything fits, can't do better
            }
            if (--nodesLeft < 0) {
                break; // safety cap; keep the best found so far
            }

            // Rebuild this combo's stats from its bits.
            int cStr = baseStr;
            int cDex = baseDex;
            int cInt = baseInt;
            int cDef = baseDef;
            int cAgi = baseAgi;
            for (int rem = combo; rem != 0; rem &= rem - 1) {
                Stats b = itemBonuses[Integer.numberOfTrailingZeros(rem)];
                cStr += b.str;
                cDex += b.dex;
                cInt += b.intel;
                cDef += b.def;
                cAgi += b.agi;
            }

            // Try adding each missing item.
            for (int slot = 0; slot < slots; slot++) {
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
                } else if (!seenLarge.add(newCombo)) {
                    continue;
                }

                Stats r = itemReqs[slot];
                if (cStr < r.str || cDex < r.dex || cInt < r.intel
                    || cDef < r.def || cAgi < r.agi) {
                    continue;
                }

                Stats ab = itemBonuses[slot];
                int nStr = cStr + ab.str;
                int nDex = cDex + ab.dex;
                int nInt = cInt + ab.intel;
                int nDef = cDef + ab.def;
                int nAgi = cAgi + ab.agi;

                if (negItems[slot]) {
                    // Cascade rule: this item lowers stats, so everything
                    // already equipped has to re-pass its requirements
                    // (not counting its own bonus).
                    boolean ok = true;
                    for (int rem = combo; rem != 0; rem &= rem - 1) {
                        int other = Integer.numberOfTrailingZeros(rem);
                        Stats or = itemReqs[other];
                        Stats ob = itemBonuses[other];
                        if (nStr - ob.str < or.str || nDex - ob.dex < or.dex
                            || nInt - ob.intel < or.intel || nDef - ob.def < or.def
                            || nAgi - ob.agi < or.agi) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        continue;
                    }
                }

                int newCount = Integer.bitCount(newCombo);
                int newTotal = nStr + nDex + nInt + nDef + nAgi;
                if (newCount > bestCount || (newCount == bestCount && newTotal > bestTotal)) {
                    bestCombo = newCombo;
                    bestCount = newCount;
                    bestTotal = newTotal;
                }
                comboStack[top++] = newCombo;
            }
        }
        seenLarge = null;
        return bestCombo;
    }

}
