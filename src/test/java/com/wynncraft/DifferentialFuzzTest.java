package com.wynncraft;

import com.wynncraft.core.SyntheticEquipment;
import com.wynncraft.core.interfaces.IAlgorithm;
import com.wynncraft.core.interfaces.IEquipment;
import com.wynncraft.core.interfaces.IPlayer;
import com.wynncraft.core.interfaces.IPlayerBuilder;
import com.wynncraft.enums.SkillPoint;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Differential fuzzing for the Closure Lattice algorithm. Runs in the default
 * suite (fixed seeds, deterministic, a few seconds); isolate it with:
 * ./gradlew test -Pcases=fuzz
 *
 * Three layers of evidence:
 * 1. A permutation oracle implementing the spec literally (search over equip
 *    orders, full invariant re-scan after every add, no memoization at all),
 *    cross-checked against a mask-memoized oracle to validate empirically
 *    that feasibility is set-determined.
 * 2. The mask-memoized oracle (still no classification / closure / greedy -
 *    none of the optimizations under test) against the algorithm on tens of
 *    thousands of random small instances.
 * 3. Agreement with three independent exact submissions (CascadeBound,
 *    Starving Goblin, Subtractive BnB) on larger instances.
 */
@Tag("fuzz")
class DifferentialFuzzTest {

    private static final SkillPoint[] SKILLS = SkillPoint.values();
    private static final int S = 5;

    // ---------------------------------------------------------------- oracles

    /** Literal spec semantics: DFS over all equip orders, no memoization. */
    private static void permutationOracle(int[][] req, int[][] bon, int[] score,
                                          int mask, int[] stats, int count, int weight, int[] best) {
        if (count > best[0] || (count == best[0] && weight > best[1])) {
            best[0] = count;
            best[1] = weight;
        }
        int n = req.length;
        for (int i = 0; i < n; i++) {
            if ((mask & (1 << i)) != 0) {
                continue;
            }
            if (!insertionOk(req[i], stats)) {
                continue;
            }
            int[] next = stats.clone();
            for (int s = 0; s < S; s++) {
                next[s] += bon[i][s];
            }
            int nextMask = mask | (1 << i);
            if (!invariantOk(req, bon, nextMask, next)) {
                continue;
            }
            permutationOracle(req, bon, score, nextMask, next, count + 1, weight + score[i], best);
        }
    }

    /** Same semantics, memoized on the equipped set (feasibility is set-determined). */
    private static void maskOracle(int[][] req, int[][] bon, int[] score,
                                   int mask, int[] stats, int count, int weight,
                                   int[] best, HashSet<Integer> visited) {
        if (count > best[0] || (count == best[0] && weight > best[1])) {
            best[0] = count;
            best[1] = weight;
        }
        int n = req.length;
        for (int i = 0; i < n; i++) {
            if ((mask & (1 << i)) != 0) {
                continue;
            }
            if (!insertionOk(req[i], stats)) {
                continue;
            }
            int[] next = stats.clone();
            for (int s = 0; s < S; s++) {
                next[s] += bon[i][s];
            }
            int nextMask = mask | (1 << i);
            if (!invariantOk(req, bon, nextMask, next)) {
                continue;
            }
            if (!visited.add(nextMask)) {
                continue;
            }
            maskOracle(req, bon, score, nextMask, next, count + 1, weight + score[i], best, visited);
        }
    }

    private static boolean insertionOk(int[] itemReq, int[] stats) {
        for (int s = 0; s < S; s++) {
            if (itemReq[s] > 0 && stats[s] < itemReq[s]) {
                return false;
            }
        }
        return true;
    }

    private static boolean invariantOk(int[][] req, int[][] bon, int mask, int[] stats) {
        for (int j = 0; j < req.length; j++) {
            if ((mask & (1 << j)) == 0) {
                continue;
            }
            for (int s = 0; s < S; s++) {
                if (req[j][s] > 0 && stats[s] < req[j][s] + bon[j][s]) {
                    return false;
                }
            }
        }
        return true;
    }

    // ------------------------------------------------------------- generation

    private static final class Instance {
        int[][] req;
        int[][] bon;
        int[] score;
        int[] base;
        IEquipment[] items;
    }

    private static Instance generate(Random rnd, int n) {
        Instance inst = new Instance();
        inst.req = new int[n][S];
        inst.bon = new int[n][S];
        inst.score = new int[n];
        inst.base = new int[S];
        inst.items = new IEquipment[n];
        for (int s = 0; s < S; s++) {
            inst.base[s] = rnd.nextInt(10) < 4 ? 0 : rnd.nextInt(15);
        }
        int mode = rnd.nextInt(4);
        for (int i = 0; i < n; i++) {
            if (i > 0 && rnd.nextInt(100) < 12) {
                // Exact duplicate: same object reference appearing twice.
                int j = rnd.nextInt(i);
                inst.req[i] = inst.req[j];
                inst.bon[i] = inst.bon[j];
                inst.score[i] = inst.score[j];
                inst.items[i] = inst.items[j];
                continue;
            }
            int[] r = new int[S];
            int[] b = new int[S];
            switch (mode) {
                case 1 -> { // chain-ish on one or two lanes
                    int lane = rnd.nextInt(S);
                    r[lane] = 1 + rnd.nextInt(12);
                    b[lane] = rnd.nextInt(9) - 3;
                    if (rnd.nextBoolean()) {
                        int lane2 = rnd.nextInt(S);
                        b[lane2] += rnd.nextInt(7) - 2;
                    }
                }
                case 2 -> { // negative-heavy
                    for (int s = 0; s < S; s++) {
                        if (rnd.nextInt(10) < 4) {
                            r[s] = 1 + rnd.nextInt(10);
                        }
                        if (rnd.nextInt(10) < 5) {
                            b[s] = -(1 + rnd.nextInt(6));
                        } else if (rnd.nextInt(10) < 3) {
                            b[s] = 1 + rnd.nextInt(8);
                        }
                    }
                }
                case 3 -> { // near-boundary around base
                    for (int s = 0; s < S; s++) {
                        if (rnd.nextInt(10) < 5) {
                            r[s] = Math.max(1, inst.base[s] + rnd.nextInt(5) - 2);
                        }
                        if (rnd.nextInt(10) < 6) {
                            b[s] = rnd.nextInt(7) - 3;
                        }
                    }
                }
                default -> { // uniform
                    for (int s = 0; s < S; s++) {
                        if (rnd.nextInt(10) < 5) {
                            r[s] = 1 + rnd.nextInt(13);
                        }
                        if (rnd.nextInt(10) < 5) {
                            b[s] = rnd.nextInt(16) - 7;
                        }
                    }
                }
            }
            int sc = 0;
            for (int s = 0; s < S; s++) {
                sc += b[s];
            }
            inst.req[i] = r;
            inst.bon[i] = b;
            inst.score[i] = sc;
            inst.items[i] = SyntheticEquipment.of(r, b);
        }
        return inst;
    }

    // -------------------------------------------------------------- execution

    private static final String[] LATTICE_NAMES = {"Closure Lattice V1", "Closure Lattice V2", "Closure Lattice V3"};

    private static AlgorithmRegistry.Entry entry(String name) {
        return AlgorithmRegistry.registry().stream()
            .filter(e -> e.name().equals(name))
            .findFirst()
            .orElseThrow(() -> new IllegalStateException("Missing algorithm: " + name));
    }

    /**
     * Runs an algorithm on the instance; returns {count, weight} and validates
     * the result shape. When the entry is Closure Lattice V1, the SWAR V2 is
     * run on the same instance as well and must produce an identical
     * (count, weight) - so every oracle check transitively covers both.
     */
    private static int[] runAlgorithm(AlgorithmRegistry.Entry entry, Instance inst, boolean validateShape) {
        int[] result = runAlgorithmOnce(entry, inst, validateShape);
        if (entry.name().equals(LATTICE_NAMES[0])) {
            for (int v = 1; v < LATTICE_NAMES.length; v++) {
                int[] other = runAlgorithmOnce(entry(LATTICE_NAMES[v]), inst, validateShape);
                if (result[0] != other[0] || result[1] != other[1]) {
                    fail(LATTICE_NAMES[v] + " diverges from V1: v1 count=" + result[0]
                        + " weight=" + result[1] + " other count=" + other[0]
                        + " weight=" + other[1] + "\n" + describe(inst));
                }
            }
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private static int[] runAlgorithmOnce(AlgorithmRegistry.Entry entry, Instance inst, boolean validateShape) {
        IPlayerBuilder<?> builder = entry.builder();
        builder.equipment(inst.items);
        for (int s = 0; s < S; s++) {
            builder.allocate(SKILLS[s], inst.base[s]);
        }
        IPlayer player = builder.build();
        IAlgorithm.Result result = ((IAlgorithm<IPlayer>) entry.algorithm()).run(player);

        int count = result.valid().size();
        int weight = 0;
        for (IEquipment item : result.valid()) {
            for (int b : item.bonuses()) {
                weight += b;
            }
        }

        if (validateShape) {
            assertEquals(inst.items.length, result.valid().size() + result.invalid().size(),
                "valid + invalid must partition the input");
            Map<IEquipment, Integer> counts = new IdentityHashMap<>();
            for (IEquipment item : inst.items) {
                counts.merge(item, 1, Integer::sum);
            }
            for (IEquipment item : result.valid()) {
                counts.merge(item, -1, Integer::sum);
            }
            for (IEquipment item : result.invalid()) {
                counts.merge(item, -1, Integer::sum);
            }
            for (Map.Entry<IEquipment, Integer> e : counts.entrySet()) {
                assertEquals(0, (int) e.getValue(), "multiset mismatch for " + e.getKey());
            }
            // The player must reflect exactly the valid items' bonuses.
            int[] expected = inst.base.clone();
            for (IEquipment item : result.valid()) {
                for (int s = 0; s < S; s++) {
                    expected[s] += item.bonuses()[s];
                }
            }
            for (int s = 0; s < S; s++) {
                assertEquals(expected[s], player.total(SKILLS[s]), "player total for " + SKILLS[s]);
            }
        }
        return new int[] {count, weight};
    }

    private static String describe(Instance inst) {
        StringBuilder sb = new StringBuilder("base=").append(java.util.Arrays.toString(inst.base));
        for (int i = 0; i < inst.items.length; i++) {
            sb.append("\n  item").append(i)
                .append(" req=").append(java.util.Arrays.toString(inst.req[i]))
                .append(" bon=").append(java.util.Arrays.toString(inst.bon[i]));
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ tests

    @Test
    void permutationOracleAgreesWithMaskOracle() {
        Random rnd = new Random(0x5EED_0001L);
        for (int iter = 0; iter < 4000; iter++) {
            int n = 1 + rnd.nextInt(5);
            Instance inst = generate(rnd, n);
            int[] bestPerm = {0, 0};
            permutationOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, bestPerm);
            int[] bestMask = {0, 0};
            maskOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, bestMask, new HashSet<>());
            if (bestPerm[0] != bestMask[0] || bestPerm[1] != bestMask[1]) {
                fail("Set-determinism violated at iter " + iter + "\n" + describe(inst));
            }
        }
    }

    @Test
    void closureLatticeMatchesOracleOnSmallInstances() {
        AlgorithmRegistry.Entry lattice = entry("Closure Lattice V1");
        Random rnd = new Random(0x5EED_0002L);
        for (int iter = 0; iter < 25000; iter++) {
            int n = 1 + rnd.nextInt(7);
            Instance inst = generate(rnd, n);
            int[] oracle = {0, 0};
            maskOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, oracle, new HashSet<>());
            int[] actual = runAlgorithm(lattice, inst, true);
            if (oracle[0] != actual[0] || oracle[1] != actual[1]) {
                fail("Oracle mismatch at iter " + iter
                    + " expected count=" + oracle[0] + " weight=" + oracle[1]
                    + " got count=" + actual[0] + " weight=" + actual[1] + "\n" + describe(inst));
            }
        }
    }

    @Test
    void closureLatticeMatchesOracleOnMediumInstances() {
        AlgorithmRegistry.Entry lattice = entry("Closure Lattice V1");
        Random rnd = new Random(0x5EED_0003L);
        for (int iter = 0; iter < 1500; iter++) {
            int n = 8 + rnd.nextInt(3);
            Instance inst = generate(rnd, n);
            int[] oracle = {0, 0};
            maskOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, oracle, new HashSet<>());
            int[] actual = runAlgorithm(lattice, inst, true);
            if (oracle[0] != actual[0] || oracle[1] != actual[1]) {
                fail("Oracle mismatch at iter " + iter
                    + " expected count=" + oracle[0] + " weight=" + oracle[1]
                    + " got count=" + actual[0] + " weight=" + actual[1] + "\n" + describe(inst));
            }
        }
    }

    @Test
    void closureLatticeAgreesWithOracleAndExactRivalsOnLargerInstances() {
        // Subtractive BnB V1 is deliberately not in the reference set: this fuzzer
        // found it under-counting (e.g. base={13,9,0,10,9}, items
        // {req [4,0,0,0,5], bon [0,4,4,0,-6]} + 2x {req [7,0,0,3,6], bon [8,-5,0,0,8]}
        // - all three are equippable in the order B,B,A but it reports only 2).
        // The mask oracle is the arbiter here instead.
        AlgorithmRegistry.Entry lattice = entry("Closure Lattice V1");
        List<AlgorithmRegistry.Entry> rivals = new ArrayList<>();
        rivals.add(entry("CascadeBound V1"));
        rivals.add(entry("Starving Goblin V2"));
        Random rnd = new Random(0x5EED_0004L);
        for (int iter = 0; iter < 3000; iter++) {
            int n = 8 + rnd.nextInt(6);
            Instance inst = generate(rnd, n);
            int[] mine = runAlgorithm(lattice, inst, true);
            int[] oracle = {0, 0};
            maskOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, oracle, new HashSet<>());
            if (oracle[0] != mine[0] || oracle[1] != mine[1]) {
                fail("Oracle mismatch at iter " + iter
                    + " expected count=" + oracle[0] + " weight=" + oracle[1]
                    + " got count=" + mine[0] + " weight=" + mine[1] + "\n" + describe(inst));
            }
            for (AlgorithmRegistry.Entry rival : rivals) {
                int[] theirs = runAlgorithm(rival, inst, false);
                if (mine[0] != theirs[0] || mine[1] != theirs[1]) {
                    fail("Disagreement with " + rival.name() + " at iter " + iter
                        + " mine count=" + mine[0] + " weight=" + mine[1]
                        + " theirs count=" + theirs[0] + " weight=" + theirs[1] + "\n" + describe(inst));
                }
            }
        }
    }

    @Test
    void hugeBranchCountsReturnFeasibleSetsQuickly() {
        // Adversarial regimes far outside game data: an impossible item plus many
        // no-requirement drain items. Exercises the >62-branch greedy fallback
        // (which must not corrupt results via mask arithmetic), the duplicate
        // canonicalization, and the node budget. The returned set must always be
        // feasible and, for these shapes, exactly optimal: every drain is
        // equippable (count = n - 1) and the blocked item is not.
        AlgorithmRegistry.Entry lattice = entry("Closure Lattice V1");
        int[][] sizes = {{24}, {64}, {65}, {80}};
        for (int[] sz : sizes) {
            int drains = sz[0];
            int n = drains + 1;
            Instance inst = new Instance();
            inst.req = new int[n][S];
            inst.bon = new int[n][S];
            inst.score = new int[n];
            inst.base = new int[] {100, 0, 0, 0, 0};
            inst.items = new IEquipment[n];
            inst.req[0] = new int[] {1000, 0, 0, 0, 0};
            inst.bon[0] = new int[S];
            inst.items[0] = SyntheticEquipment.of(inst.req[0], inst.bon[0]);
            for (int i = 1; i < n; i++) {
                inst.req[i] = new int[S];
                // Heterogeneous drains (distinct lanes) so subsets do not collapse
                // by duplicate canonicalization alone.
                int[] b = new int[S];
                b[1 + (i % 4)] = -1;
                b[0] = (i % 3 == 0) ? -1 : 0;
                inst.bon[i] = b;
                int sc = 0;
                for (int s = 0; s < S; s++) {
                    sc += b[s];
                }
                inst.score[i] = sc;
                inst.items[i] = SyntheticEquipment.of(inst.req[i], inst.bon[i]);
            }
            long start = System.nanoTime();
            int[] actual = runAlgorithm(lattice, inst, true);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;
            assertEquals(drains, actual[0], "all drains must be valid for " + drains + " drains");
            assertTrue(elapsedMs < 5_000, "bounded latency, took " + elapsedMs + "ms for " + drains + " drains");
        }
    }

    @Test
    void scratchBuffersSurviveShrinkingAndGrowingInputs() {
        // One shared instance across wildly varying n exercises stale-state bugs.
        AlgorithmRegistry.Entry lattice = entry("Closure Lattice V1");
        Random rnd = new Random(0x5EED_0005L);
        int[] sizes = {13, 1, 9, 2, 12, 3, 11, 1, 10, 4};
        for (int round = 0; round < 300; round++) {
            for (int n : sizes) {
                Instance inst = generate(rnd, n);
                int[] oracle = {0, 0};
                maskOracle(inst.req, inst.bon, inst.score, 0, inst.base.clone(), 0, 0, oracle, new HashSet<>());
                int[] actual = runAlgorithm(lattice, inst, true);
                if (oracle[0] != actual[0] || oracle[1] != actual[1]) {
                    fail("Stale-state mismatch at round " + round + " n=" + n
                        + " expected count=" + oracle[0] + " weight=" + oracle[1]
                        + " got count=" + actual[0] + " weight=" + actual[1] + "\n" + describe(inst));
                }
            }
        }
        assertTrue(true);
    }

}
