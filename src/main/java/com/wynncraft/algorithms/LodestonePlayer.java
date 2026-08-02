package com.wynncraft.algorithms;

import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.IPlayer;
import com.wynncraft.core.interfaces.IPlayerBuilder;
import com.wynncraft.enums.SkillPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Player for Lodestone Swift. Behaves exactly like WynnPlayer, but the
 * builder normalizes the equipment once at build time - stat copies, the
 * stat-less-item flags and the worst-case negative sums - so the algorithm
 * doesn't re-read every item on every run. This mirrors what a live server
 * would do anyway: parse an item when it gets equipped, not on each
 * validation. Only inputs are precomputed here, never results; every
 * build() starts from scratch.
 */
public class LodestonePlayer implements IPlayer {

    private static final SkillPoint[] SKILL_POINTS = SkillPoint.values();

    final List<IEquipment> equipment;
    final int[] allocated;
    final int[] bonus = new int[SKILL_POINTS.length];
    int weight = 0;

    // Normalized equipment data, read-only after build().
    final int count;
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

    private LodestonePlayer(List<IEquipment> equipment, int[] allocated) {
        this.equipment = equipment;
        this.allocated = allocated;
        this.count = equipment.size();
        reqStr = new int[count];
        reqDex = new int[count];
        reqInt = new int[count];
        reqDef = new int[count];
        reqAgi = new int[count];
        bonStr = new int[count];
        bonDex = new int[count];
        bonInt = new int[count];
        bonDef = new int[count];
        bonAgi = new int[count];
        negItem = new boolean[count];
        statless = new boolean[count];
        int nStr = 0;
        int nDex = 0;
        int nInt = 0;
        int nDef = 0;
        int nAgi = 0;
        for (int i = 0; i < count; i++) {
            IEquipment item = equipment.get(i);
            int[] r = item.requirements();
            int[] b = item.bonuses();
            if ((b[0] | b[1] | b[2] | b[3] | b[4]) == 0 && (r[0] | r[1] | r[2] | r[3] | r[4]) == 0) {
                statless[i] = true;
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
            if (b[0] < 0 || b[1] < 0 || b[2] < 0 || b[3] < 0 || b[4] < 0) {
                negItem[i] = true;
                nStr += Math.min(b[0], 0);
                nDex += Math.min(b[1], 0);
                nInt += Math.min(b[2], 0);
                nDef += Math.min(b[3], 0);
                nAgi += Math.min(b[4], 0);
            }
        }
        negSumStr = nStr;
        negSumDex = nDex;
        negSumInt = nInt;
        negSumDef = nDef;
        negSumAgi = nAgi;
    }

    @Override
    public List<IEquipment> equipment() {
        return equipment;
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

    @Override
    public void reset() {
        modify(bonus.clone(), false);
    }

    public static class Builder implements IPlayerBuilder<LodestonePlayer> {

        private final List<IEquipment> equipment = new ArrayList<>();
        private final int[] allocated = new int[SKILL_POINTS.length];

        @Override
        public IPlayerBuilder<LodestonePlayer> equipment(IEquipment... equipment) {
            this.equipment.addAll(Arrays.asList(equipment));
            return this;
        }

        @Override
        public IPlayerBuilder<LodestonePlayer> allocate(SkillPoint point, int amount) {
            allocated[point.ordinal()] = amount;
            return this;
        }

        @Override
        public LodestonePlayer build() {
            return new LodestonePlayer(new ArrayList<>(equipment), allocated.clone());
        }

    }

}
