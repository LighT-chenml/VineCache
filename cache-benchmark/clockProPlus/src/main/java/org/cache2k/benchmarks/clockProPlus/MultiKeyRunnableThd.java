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

    // Multi-key specific fields
    private final List<List<String>> groupedWorkloadKeys = new ArrayList<>();
    private final int groupSize;
    private final List<Integer> hitCountDistribution = new ArrayList<>();
    private final IMultiKeyCache multiKeyCache;
    
    // Benchmark configuration
    private final int benchmarkWorkloadSize;
    private int warmUpSize;

    /**
     * Creates a new multi-key benchmark thread.
     * 
     * @param name Thread name for identification
     * @param cache Multi-key cache instance to benchmark
     * @param workloadKeys List of keys to use in benchmark
     * @param groupSize Number of keys to group together for multi-key operations
     * @param benchmarkWorkloadSize Size of the benchmark workload
     */
    MultiKeyRunnableThd(String name, ISimpleCache cache, List<String> workloadKeys,
                       int groupSize, int benchmarkWorkloadSize) {
        super(name, cache, workloadKeys, benchmarkWorkloadSize);
        
        this.groupSize = groupSize;
        this.benchmarkWorkloadSize = benchmarkWorkloadSize;
        this.multiKeyCache = (IMultiKeyCache) cache;
        
        // Initialize hit count distribution tracking
        for (int i = 0; i <= groupSize; i++) {
            hitCountDistribution.add(0);
        }
        
        groupWorkloadKeys();
        System.out.println("Created multi-key benchmark thread: " + name + 
                          " with " + groupedWorkloadKeys.size() + " groups, group size: " + groupSize);
    }

    /**
     * Groups the workload keys into chunks of specified group size.
     */
    private void groupWorkloadKeys() {
        for (int i = 0; i < workloadKeys.size(); i += groupSize) {
            List<String> currentGroup = new ArrayList<>();
            
            // Add keys to current group (up to groupSize)
            for (int j = i; j < Math.min(i + groupSize, workloadKeys.size()); j++) {
                currentGroup.add(workloadKeys.get(j));
            }
            
            groupedWorkloadKeys.add(currentGroup);
        }
        
        System.out.println("Grouped " + workloadKeys.size() + " keys into " + 
                          groupedWorkloadKeys.size() + " groups");
    }

    /**
     * Warms up the cache using sequential access pattern for grouped keys.
     * 
     * @param warmUpPercentage Percentage of workload to use for warm-up (0-100)
     */
    @Override
    public void warmUpTheCache(int warmUpPercentage) {
        System.out.println("Starting sequential warm-up with " + warmUpPercentage + "% of workload");
        
        warmUpSize = benchmarkWorkloadSize * warmUpPercentage / 100;
        
        for (int i = 0; i < warmUpSize && i < groupedWorkloadKeys.size(); i++) {
            multiKeyCache.request(groupedWorkloadKeys.get(i));
        }
        
        System.out.println("Sequential warm-up completed");
    }

    /**
     * Warms up the cache using random access pattern for grouped keys.
     * 
     * @param multiplier Multiple of cache size to use for warm-up
     */
    public void warmUpTheCacheRandomly(int multiplier) {
        warmUpSize = multiKeyCache.getCacheSize() * multiplier;
        
        System.out.println("Starting random warm-up with " + multiplier + "x cache size");
        System.out.println("Warm-up size: " + warmUpSize + 
                          ", Available groups: " + groupedWorkloadKeys.size() * groupSize);
        
        if (warmUpSize >= groupedWorkloadKeys.size() * groupSize) {
            // Use sequential access if warm-up size exceeds available keys
            System.out.println("Using sequential access (warm-up size >= available keys)");
            int index = 0;
            while (warmUpSize > 0 && index < groupedWorkloadKeys.size()) {
                multiKeyCache.request(groupedWorkloadKeys.get(index));
                index++;
                warmUpSize -= groupSize;
            }
        } else {
            // Use random access
            Random random = new Random(warmUpSize); // Use warm-up size as seed for reproducibility
            int maxIndex = groupedWorkloadKeys.size() - 1;
            
            while (warmUpSize > 0) {
                int randomIndex = random.nextInt(maxIndex);
                multiKeyCache.request(groupedWorkloadKeys.get(randomIndex));
                warmUpSize -= groupSize;
            }
        }
        
        System.out.println("Random warm-up completed");
    }

    /**
     * Analyzes workload patterns to understand key reuse within sliding windows.
     * This method examines temporal locality in the grouped workload.
     */
    void analyzeWorkload() {
        System.out.println("Starting workload analysis...");
        
        HashMap<String, Integer> keyCountMap = new HashMap<>();
        int[] hitCounts = new int[3]; // Track hits at different frequency levels
        
        // Process workload in reverse order with sliding window
        for (int i = groupedWorkloadKeys.size() - 1; i >= 0; i--) {
            List<String> currentGroup = groupedWorkloadKeys.get(i);
            
            // Count hits for current group based on previous occurrences
            for (String key : currentGroup) {
                if (keyCountMap.containsKey(key)) {
                    int frequency = keyCountMap.get(key);
                    if (frequency == 1) hitCounts[0]++;
                    else if (frequency == 2) hitCounts[1]++;
                    else hitCounts[2]++;
                }
            }
            
            // Update key frequencies
            for (String key : currentGroup) {
                keyCountMap.put(key, keyCountMap.getOrDefault(key, 0) + 1);
            }
            
            // Remove keys outside sliding window (10000 groups back)
            if (i + 10000 < groupedWorkloadKeys.size()) {
                List<String> oldGroup = groupedWorkloadKeys.get(i + 10000);
                for (String key : oldGroup) {
                    int count = keyCountMap.get(key);
                    keyCountMap.put(key, count - 1);
                }
            }
        }
        
        // Output analysis results as ratios
        int totalGroups = groupedWorkloadKeys.size();
        System.out.printf("Hit ratio (freq=1): %.6f%n", (double) hitCounts[0] / totalGroups / 26);
        System.out.printf("Hit ratio (freq=2): %.6f%n", (double) hitCounts[1] / totalGroups / 26);
        System.out.printf("Hit ratio (freq>=3): %.6f%n", (double) hitCounts[2] / totalGroups / 26);
        System.out.println("Workload analysis completed");
    }

    /**
     * Outputs the cumulative distribution function of hit counts.
     */
    void outputHitCountCDF() {
        System.out.println("Hit Count CDF:");
        
        int totalSamples = hitCountDistribution.stream().mapToInt(Integer::intValue).sum();
        double cumulativeProbability = 0.0;
        
        // Print CDF in descending order
        for (int i = groupSize; i >= 0; i--) {
            cumulativeProbability += (double) hitCountDistribution.get(i) / totalSamples;
            System.out.printf("P(hits >= %d) = %.4f%n", i, cumulativeProbability);
        }
    }

    /**
     * Main benchmark execution logic for multi-key operations.
     * Processes grouped keys and records hit/miss patterns.
     */
    @Override
    public void run() {
        System.out.println("Starting multi-key benchmark thread: " + threadName);
        
        // Calculate the starting index for benchmark portion of workload
        int startIndex = groupedWorkloadKeys.size() - benchmarkWorkloadSize;
        
        // Process each group in the benchmark workload
        for (int i = startIndex; i < groupedWorkloadKeys.size(); i++) {
            List<Boolean> groupHitResults = multiKeyCache.request(groupedWorkloadKeys.get(i));
            
            // Count hits in current group and record individual results
            int groupHitCount = 0;
            for (Boolean isHit : groupHitResults) {
                recordHitMiss.add(isHit);
                if (isHit) {
                    groupHitCount++;
                }
            }
            
            // Update hit count distribution
            int currentCount = hitCountDistribution.get(groupHitCount);
            hitCountDistribution.set(groupHitCount, currentCount + 1);
        }
        
        System.out.println("Thread " + threadName + " completed multi-key benchmark");
    }
}
