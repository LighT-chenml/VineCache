package org.cache2k.benchmarks.clockProPlus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A thread that executes cache benchmark by processing multiple keys at a time.
 * This class extends SingleKeyRunnableThd to handle grouped key operations for multi-key cache testing.
 */
class MultiKeyRunnableThd extends SingleKeyRunnableThd {

    // Multi-key Workload Details
    private final List<List<String>> groupedWorkloadKeys = new ArrayList<>();
    private final int groupSize;
    private final IMultiKeyCache multiKeyCache;
    
    // Warm-up and Execution State
    private int warmUpCount;

    /**
     * Initializes a multi-key benchmark thread.
     *
     * @param name                  The name of the thread.
     * @param cache                 The cache implementation to benchmark.
     * @param workloadKeys          The sequence of keys for the benchmark.
     * @param groupSize             The size of each key group.
     * @param benchmarkWorkloadSize The number of groups to process in the evaluation phase.
     */
    MultiKeyRunnableThd(String name, ISimpleCache cache, List<String> workloadKeys,
                       int groupSize, int benchmarkWorkloadSize) {
        super(name, cache, workloadKeys, benchmarkWorkloadSize);
        
        this.groupSize = groupSize;
        this.multiKeyCache = (IMultiKeyCache) cache;

        syncDistributionTracking(groupSize);
        groupWorkloadKeys();
        
        System.out.println("[Benchmark] Created multi-key thread: " + name + " (GroupSize=" + groupSize + ")");
    }

    private void syncDistributionTracking(int size) {
        if (hitCountDistribution.size() != size + 1) {
            hitCountDistribution.clear();
            for (int i = 0; i <= size; i++) {
                hitCountDistribution.add(0);
            }
        }
    }

    /**
     * Groups workload keys into chunks of the specified size.
     */
    private void groupWorkloadKeys() {
        for (int i = 0; i < workloadKeys.size(); i += groupSize) {
            int end = Math.min(i + groupSize, workloadKeys.size());
            groupedWorkloadKeys.add(new ArrayList<>(workloadKeys.subList(i, end)));
        }
        System.out.println("[Workload] Grouped " + workloadKeys.size() + " keys into " + groupedWorkloadKeys.size() + " groups.");
    }

    /**
     * Performs a sequential warm-up of the multi-key cache.
     *
     * @param warmUpPercentage Percentage of benchmark workload to use (0-100).
     */
    @Override
    public void warmUpTheCache(int warmUpPercentage) {
        warmUpCount = (benchmarkWorkloadSize * warmUpPercentage / 100);
        System.out.println("[Warm-up] Multi-key sequential start (" + warmUpPercentage + "%, size=" + warmUpCount + " groups)");

        for (int i = 0; i < warmUpCount && i < groupedWorkloadKeys.size(); i++) {
            multiKeyCache.request(groupedWorkloadKeys.get(i));
        }
        System.out.println("[Warm-up] Multi-key sequential completed.");
    }

    /**
     * Performs a random warm-up of the multi-key cache.
     *
     * @param multiplier Number of times the cache size to request.
     */
    public void warmUpTheCacheRandomly(int multiplier) {
        warmUpCount = multiKeyCache.getCacheSize() * multiplier;
        System.out.println("[Warm-up] Multi-key random start (" + multiplier + "x cache size, count=" + warmUpCount + " keys)");

        int groupsToRequest = warmUpCount / groupSize;
        if (groupsToRequest >= groupedWorkloadKeys.size()) {
            System.out.println("[Warm-up] Switching to sequential (count exceeds available keys).");
            groupedWorkloadKeys.forEach(multiKeyCache::request);
        } else {
            Random random = new Random(warmUpCount);
            int bound = groupedWorkloadKeys.size();
            for (int i = 0; i < groupsToRequest; i++) {
                multiKeyCache.request(groupedWorkloadKeys.get(random.nextInt(bound)));
            }
        }
        System.out.println("[Warm-up] Multi-key random completed.");
    }

    /**
     * Analyzes workload patterns to understand key reuse within a sliding window.
     */
    void analyzeWorkload() {
        System.out.println("[Analysis] Starting workload temporal locality analysis...");
        
        HashMap<String, Integer> keyFrequencyMap = new HashMap<>();
        int[] frequencyHits = new int[3]; // [hits_at_f1, hits_at_f2, hits_at_f3+]
        
        for (int i = groupedWorkloadKeys.size() - 1; i >= 0; i--) {
            List<String> group = groupedWorkloadKeys.get(i);
            
            for (String key : group) {
                int freq = keyFrequencyMap.getOrDefault(key, 0);
                if (freq > 0) {
                    frequencyHits[Math.min(freq - 1, 2)]++;
                }
                keyFrequencyMap.put(key, freq + 1);
            }
            
            // Sliding window cleanup (last 10000 groups)
            if (i + 10000 < groupedWorkloadKeys.size()) {
                for (String key : groupedWorkloadKeys.get(i + 10000)) {
                    keyFrequencyMap.computeIfPresent(key, (k, v) -> v - 1);
                }
            }
        }
        
        double total = (double) groupedWorkloadKeys.size() * groupSize;
        System.out.printf("[Analysis] Hit Ratios: F1=%.6f, F2=%.6f, F3+=%.6f%n", 
            frequencyHits[0]/total, frequencyHits[1]/total, frequencyHits[2]/total);
    }

    /**
     * Prints the Cumulative Distribution Function (CDF) of group hit counts.
     */
    void outputHitCountCDF() {
        int totalGroups = hitCountDistribution.stream().mapToInt(Integer::intValue).sum();
        if (totalGroups == 0) return;

        System.out.println("[Stats] Multi-key Hit Count CDF:");
        double cumulativeProb = 0.0;
        for (int i = groupSize; i >= 0; i--) {
            cumulativeProb += (double) hitCountDistribution.get(i) / totalGroups;
            System.out.printf("  P(hits >= %d) = %.4f%n", i, cumulativeProb);
        }
    }

    @Override
    public void run() {
        System.out.println("[Execution] Multi-key thread " + threadName + " started.");
        
        int startIndex = Math.max(0, groupedWorkloadKeys.size() - benchmarkWorkloadSize);
        
        for (int i = startIndex; i < groupedWorkloadKeys.size(); i++) {
            List<Boolean> groupResults = multiKeyCache.request(groupedWorkloadKeys.get(i));
            
            int groupHits = 0;
            for (Boolean isHit : groupResults) {
                recordHitMiss.add(isHit);
                if (isHit) {
                    groupHits++;
                    localHitCount++;
                }
            }
            
            hitCountDistribution.set(groupHits, hitCountDistribution.get(groupHits) + 1);
        }
        
        System.out.println("[Execution] Multi-key thread " + threadName + " completed.");
    }
}
