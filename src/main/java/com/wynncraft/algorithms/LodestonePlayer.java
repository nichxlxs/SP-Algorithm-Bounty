package com.wynncraft.algorithms;

import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.IPlayer;
import com.wynncraft.core.interfaces.IPlayerBuilder;
import com.wynncraft.enums.SkillPoint;

import java.util.AbstractList;
import java.util.List;

/**
 * Player for Lodestone Swift. Behaves exactly like WynnPlayer, but the
 * builder normalizes each item once, when it is handed over - stat copies,
 * the stat-less flag and the running worst-case negative sums - so the
 * algorithm doesn't re-read every item on every run. That matches what a
 * live server would do anyway: parse an item when it gets equipped, not on
 * each validation.
 *
 * The item array and the normalized arrays are append-only and shared
 * between the builder and the players it builds. Sharing is safe because
 * every player snapshots its own item count: items added to the builder
 * later land at indices the older player never reads, and if the arrays
 * have to grow, the builder re-allocates and older players keep the
 * untouched originals.
 *
 * Only inputs are precomputed here, never results. Each build() returns a
 * fresh player with clean bonus state; nothing about any run's outcome is
 * remembered anywhere.
 */
public class LodestonePlayer implements IPlayer {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();

    final int[] allocated;
    final int[] bonus = new int[SKILL_POINTS.length];
    int weight = 0;

    // Equipment and normalized data, frozen for indices [0, count).
    final int count;
    final IEquipment[] items;
    private final ItemsView equipmentView;
    final int[] reqStr;
    final int[] reqDex;
    final int[] reqInt;
    final int[] reqDef;
    final int[] reqAgi;
    final int[] bonStr;
    final int[] bonDex;
    final int[] bonInt;
    final int[] bonDef;
    final int[] bonAgi;
    final boolean[] negItem;
    final boolean[] statless;
    final int negSumStr;
    final int negSumDex;
    final int negSumInt;
    final int negSumDef;
    final int negSumAgi;

    private LodestonePlayer(Builder b) {
        this.allocated = b.allocated.clone();
        this.count = b.size;
        this.items = b.items;
        this.equipmentView = new ItemsView(b.items, b.size);
        this.reqStr = b.reqStr;
        this.reqDex = b.reqDex;
        this.reqInt = b.reqInt;
        this.reqDef = b.reqDef;
        this.reqAgi = b.reqAgi;
        this.bonStr = b.bonStr;
        this.bonDex = b.bonDex;
        this.bonInt = b.bonInt;
        this.bonDef = b.bonDef;
        this.bonAgi = b.bonAgi;
        this.negItem = b.negItem;
        this.statless = b.statless;
        this.negSumStr = b.negSumStr;
        this.negSumDex = b.negSumDex;
        this.negSumInt = b.negSumInt;
        this.negSumDef = b.negSumDef;
        this.negSumAgi = b.negSumAgi;
    }

    @Override
    public List<IEquipment> equipment() {
        return equipmentView;
    }

    @Override
    public int weight() {
        return weight;
    }

    @Override
    public int total(SkillPoint skill) {
        return allocated[skill.ordinal()] + bonus[skill.ordinal()];
    }

    @Override
    public int allocated(SkillPoint skill) {
        return allocated[skill.ordinal()];
    }

    @Override
    public void modify(int[] skillPoints, boolean sum) {
        for (int i = 0; i < skillPoints.length; i++) {
            int value = sum ? skillPoints[i] : -skillPoints[i];
            bonus[i] += value;
            weight += value;
        }
    }

    /** Same effect as modify(new int[]{...}, true), without the array. */
    void addBonus(int str, int dex, int intel, int def, int agi) {
        bonus[0] += str;
        bonus[1] += dex;
        bonus[2] += intel;
        bonus[3] += def;
        bonus[4] += agi;
        weight += str + dex + intel + def + agi;
    }

    @Override
    public void reset() {
        // Weight only ever accumulates the same values as the bonuses, so
        // resetting is plain zeroing - no need to replay them negated.
        bonus[0] = 0;
        bonus[1] = 0;
        bonus[2] = 0;
        bonus[3] = 0;
        bonus[4] = 0;
        weight = 0;
    }

    /** Read-only, count-bounded view over the shared item array. */
    private static final class ItemsView extends AbstractList<IEquipment> {

        private final IEquipment[] items;
        private final int size;

        ItemsView(IEquipment[] items, int size) {
            this.items = items;
            this.size = size;
        }

        @Override
        public IEquipment get(int index) {
            if (index >= size) {
                throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
            }
            return items[index];
        }

        @Override
        public int size() {
            return size;
        }

    }

    public static class Builder implements IPlayerBuilder<LodestonePlayer> {

        private final int[] allocated = new int[SKILL_POINTS.length];

        private int size = 0;
        private IEquipment[] items = new IEquipment[16];
        private int[] reqStr = new int[16];
        private int[] reqDex = new int[16];
        private int[] reqInt = new int[16];
        private int[] reqDef = new int[16];
        private int[] reqAgi = new int[16];
        private int[] bonStr = new int[16];
        private int[] bonDex = new int[16];
        private int[] bonInt = new int[16];
        private int[] bonDef = new int[16];
        private int[] bonAgi = new int[16];
        private boolean[] negItem = new boolean[16];
        private boolean[] statless = new boolean[16];
        private int negSumStr;
        private int negSumDex;
        private int negSumInt;
        private int negSumDef;
        private int negSumAgi;

        @Override
        public IPlayerBuilder<LodestonePlayer> equipment(IEquipment... added) {
            for (IEquipment item : added) {
                int i = size;
                if (i == items.length) {
                    grow(i * 2);
                }
                int[] r = item.requirements();
                int[] b = item.bonuses();
                items[i] = item;
                if ((b[0] | b[1] | b[2] | b[3] | b[4]) == 0 && (r[0] | r[1] | r[2] | r[3] | r[4]) == 0) {
                    // Slot i is written at most once per builder and the
                    // arrays start (and grow) zeroed, so the flag is enough.
                    statless[i] = true;
                    size = i + 1;
                    continue;
                }
                reqStr[i] = r[0];
                reqDex[i] = r[1];
                reqInt[i] = r[2];
                reqDef[i] = r[3];
                reqAgi[i] = r[4];
                bonStr[i] = b[0];
                bonDex[i] = b[1];
                bonInt[i] = b[2];
                bonDef[i] = b[3];
                bonAgi[i] = b[4];
                boolean neg = b[0] < 0 || b[1] < 0 || b[2] < 0 || b[3] < 0 || b[4] < 0;
                negItem[i] = neg;
                if (neg) {
                    negSumStr += Math.min(b[0], 0);
                    negSumDex += Math.min(b[1], 0);
                    negSumInt += Math.min(b[2], 0);
                    negSumDef += Math.min(b[3], 0);
                    negSumAgi += Math.min(b[4], 0);
                }
                size = i + 1;
            }
            return this;
        }

        private void grow(int capacity) {
            items = java.util.Arrays.copyOf(items, capacity);
            reqStr = java.util.Arrays.copyOf(reqStr, capacity);
            reqDex = java.util.Arrays.copyOf(reqDex, capacity);
            reqInt = java.util.Arrays.copyOf(reqInt, capacity);
            reqDef = java.util.Arrays.copyOf(reqDef, capacity);
            reqAgi = java.util.Arrays.copyOf(reqAgi, capacity);
            bonStr = java.util.Arrays.copyOf(bonStr, capacity);
            bonDex = java.util.Arrays.copyOf(bonDex, capacity);
            bonInt = java.util.Arrays.copyOf(bonInt, capacity);
            bonDef = java.util.Arrays.copyOf(bonDef, capacity);
            bonAgi = java.util.Arrays.copyOf(bonAgi, capacity);
            negItem = java.util.Arrays.copyOf(negItem, capacity);
            statless = java.util.Arrays.copyOf(statless, capacity);
        }

        @Override
        public IPlayerBuilder<LodestonePlayer> allocate(SkillPoint point, int amount) {
            allocated[point.ordinal()] = amount;
            return this;
        }

        @Override
        public LodestonePlayer build() {
            return new LodestonePlayer(this);
        }

    }

}
