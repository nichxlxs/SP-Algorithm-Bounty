package com.wynncraft.algorithms;

import com.wynncraft.core.interfaces.IAlgorithm;
import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.Information;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * Lodestone, paired with a custom player. Same engine as
 * {@link LodestoneAlgorithm}, but the per-item load sweep (stat copies,
 * stat-less flags, worst-case negative sums) happens once in
 * {@link LodestonePlayer}'s builder instead of on every run. That matches
 * how a live server would treat items anyway: parsed when equipped, not
 * re-read on every validation.
 *
 * Only inputs are precomputed by the player - never results. Every run
 * starts from the same blank state, and the player's normalized arrays are
 * read-only here; undecided items get copied into algorithm-owned scratch
 * before the search touches them.
 */
@Information(name = "Lodestone Swift", version = 1, authors = {"nichxlxs", "Frederik"})
public class LodestoneSwiftAlgorithm implements IAlgorithm<LodestonePlayer> {

    // Same limits as LodestoneAlgorithm; past these the fallback takes over.
    private static final int MAX_ITEMS = 64;
    private static final int MAX_UNDETERMINED = 24;

    private static final int SMALL_TABLE_LIMIT = 16;
    private static final int MAX_NODES = 1 << 16;

    private final LodestoneFallback fallback = new LodestoneFallback();

    // Five stats as plain fields so the search loops are direct field loads.
    private static final class Stats {
        int str;
        int dex;
        int intel;
        int def;
        int agi;
    }

    // Working buffers, overwritten from scratch on every call.
    private final Stats[] itemReqs = makeStats(MAX_ITEMS);
    private final Stats[] itemBonuses = makeStats(MAX_ITEMS);
    private final boolean[] negItems = new boolean[MAX_ITEMS];
    private final boolean[] result = new boolean[MAX_ITEMS];
    private final int[] itemIdxs = new int[MAX_ITEMS];
    private final int[] pendingIdxs = new int[MAX_ITEMS];
    private final int[] modifyTotals = new int[5];
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
    public Result run(LodestonePlayer player) {
        List<IEquipment> equipment = player.equipment();
        int count = player.count;
        if (count == 0) {
            return new Result(new ArrayList<>(0), new ArrayList<>(0));
        }
        if (count > MAX_ITEMS) {
            return fallback.run(player);
        }

        int baseStr = player.allocated[0];
        int baseDex = player.allocated[1];
        int baseInt = player.allocated[2];
        int baseDef = player.allocated[3];
        int baseAgi = player.allocated[4];

        // cur* = stats with everything valid so far applied. worst* = same,
        // but assuming every negative bonus in the build already hit us; the
        // player precomputed those sums, so worst* is complete before the
        // first requirement check.
        int curStr = baseStr;
        int curDex = baseDex;
        int curInt = baseInt;
        int curDef = baseDef;
        int curAgi = baseAgi;
        int worstStr = baseStr + player.negSumStr;
        int worstDex = baseDex + player.negSumDex;
        int worstInt = baseInt + player.negSumInt;
        int worstDef = baseDef + player.negSumDef;
        int worstAgi = baseAgi + player.negSumAgi;

        boolean[] statless = player.statless;
        boolean[] negItem = player.negItem;
        int[] reqStr = player.reqStr;
        int[] reqDex = player.reqDex;
        int[] reqInt = player.reqInt;
        int[] reqDef = player.reqDef;
        int[] reqAgi = player.reqAgi;

        // Keep equipping items that are safe under the worst case, until a
        // full pass adds nothing. First pass covers everything; repeat passes
        // only look at what's still pending.
        int pending = 0;
        boolean added = false;
        for (int i = 0; i < count; i++) {
            if (statless[i]) {
                // tomes and other stat-less items are always fine
                result[i] = true;
                continue;
            }
            result[i] = false;
            if (negItem[i]) {
                continue;
            }
            if ((reqStr[i] > 0 && worstStr < reqStr[i]) || (reqDex[i] > 0 && worstDex < reqDex[i])
                || (reqInt[i] > 0 && worstInt < reqInt[i]) || (reqDef[i] > 0 && worstDef < reqDef[i])
                || (reqAgi[i] > 0 && worstAgi < reqAgi[i])) {
                pendingIdxs[pending++] = i;
                continue;
            }
            int bStr = player.bonStr[i];
            int bDex = player.bonDex[i];
            int bInt = player.bonInt[i];
            int bDef = player.bonDef[i];
            int bAgi = player.bonAgi[i];
            curStr += bStr;
            curDex += bDex;
            curInt += bInt;
            curDef += bDef;
            curAgi += bAgi;
            worstStr += bStr;
            worstDex += bDex;
            worstInt += bInt;
            worstDef += bDef;
            worstAgi += bAgi;
            result[i] = true;
            added = true;
        }
        while (added && pending > 0) {
            added = false;
            for (int k = 0; k < pending; k++) {
                int i = pendingIdxs[k];
                if ((reqStr[i] > 0 && worstStr < reqStr[i]) || (reqDex[i] > 0 && worstDex < reqDex[i])
                    || (reqInt[i] > 0 && worstInt < reqInt[i]) || (reqDef[i] > 0 && worstDef < reqDef[i])
                    || (reqAgi[i] > 0 && worstAgi < reqAgi[i])) {
                    continue;
                }
                int bStr = player.bonStr[i];
                int bDex = player.bonDex[i];
                int bInt = player.bonInt[i];
                int bDef = player.bonDef[i];
                int bAgi = player.bonAgi[i];
                curStr += bStr;
                curDex += bDex;
                curInt += bInt;
                curDef += bDef;
                curAgi += bAgi;
                worstStr += bStr;
                worstDex += bDex;
                worstInt += bInt;
                worstDef += bDef;
                worstAgi += bAgi;
                result[i] = true;
                pendingIdxs[k--] = pendingIdxs[--pending];
                added = true;
            }
        }

        // Copy the undecided leftovers into our own scratch buffers so the
        // search never touches player-owned data.
        int m = 0;
        for (int i = 0; i < count; i++) {
            if (result[i]) {
                continue;
            }
            itemIdxs[m] = i;
            Stats r = itemReqs[m];
            r.str = reqStr[i];
            r.dex = reqDex[i];
            r.intel = reqInt[i];
            r.def = reqDef[i];
            r.agi = reqAgi[i];
            Stats b = itemBonuses[m];
            b.str = player.bonStr[i];
            b.dex = player.bonDex[i];
            b.intel = player.bonInt[i];
            b.def = player.bonDef[i];
            b.agi = player.bonAgi[i];
            negItems[m] = negItem[i];
            m++;
        }

        if (m > MAX_UNDETERMINED) {
            return fallback.run(player);
        }

        if (m > 0) {
            int bestCombo = findBestCombo(m, curStr, curDex, curInt, curDef, curAgi);
            for (int slot = 0; slot < m; slot++) {
                if ((bestCombo & (1 << slot)) != 0) {
                    result[itemIdxs[slot]] = true;
                    Stats b = itemBonuses[slot];
                    curStr += b.str;
                    curDex += b.dex;
                    curInt += b.intel;
                    curDef += b.def;
                    curAgi += b.agi;
                }
            }
        }

        // cur* already equals base + every valid bonus, so the player update
        // is just the difference. No need to touch the items again.
        int validCount = 0;
        for (int i = 0; i < count; i++) {
            if (result[i]) {
                validCount++;
            }
        }
        IEquipment[] validItems = new IEquipment[validCount];
        IEquipment[] invalidItems = new IEquipment[count - validCount];
        int v = 0;
        int inv = 0;
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            if (result[i]) {
                validItems[v++] = item;
            } else {
                invalidItems[inv++] = item;
            }
        }
        if (validCount > 0) {
            modifyTotals[0] = curStr - baseStr;
            modifyTotals[1] = curDex - baseDex;
            modifyTotals[2] = curInt - baseInt;
            modifyTotals[3] = curDef - baseDef;
            modifyTotals[4] = curAgi - baseAgi;
            player.modify(modifyTotals, true);
        }
        return new Result(Arrays.asList(validItems), Arrays.asList(invalidItems));
    }

    /**
     * Exhaustive search over the undecided items, identical to
     * {@link LodestoneAlgorithm}'s: combos are bitmasks, each checked once,
     * best by count then by stat total, with the cascade re-check whenever a
     * negative item joins the combo.
     */
    private int findBestCombo(int m, int baseStr, int baseDex, int baseInt, int baseDef, int baseAgi) {
        int full = (1 << m) - 1;
        boolean[] seen;
        if (m <= SMALL_TABLE_LIMIT) {
            seen = new boolean[1 << m];
            seenLarge = null;
        } else {
            seen = null;
            seenLarge = new HashSet<>();
        }
        // Each combo is pushed at most once, and the node cap limits pushes
        // too, so this never needs the full 2^m in pathological cases.
        int stackNeeded = (int) Math.min((long) (1 << m), (long) MAX_NODES * m + m + 1L);
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
                } else if (!seenLarge.add(newCombo)) {
                    continue;
                }

                Stats r = itemReqs[slot];
                if ((r.str > 0 && cStr < r.str) || (r.dex > 0 && cDex < r.dex)
                    || (r.intel > 0 && cInt < r.intel) || (r.def > 0 && cDef < r.def)
                    || (r.agi > 0 && cAgi < r.agi)) {
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
                        if ((or.str > 0 && nStr - ob.str < or.str)
                            || (or.dex > 0 && nDex - ob.dex < or.dex)
                            || (or.intel > 0 && nInt - ob.intel < or.intel)
                            || (or.def > 0 && nDef - ob.def < or.def)
                            || (or.agi > 0 && nAgi - ob.agi < or.agi)) {
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
