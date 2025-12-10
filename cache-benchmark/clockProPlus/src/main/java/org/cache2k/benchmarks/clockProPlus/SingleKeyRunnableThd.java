package org.cache2k.benchmarks.clockProPlus;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A thread that executes cache benchmark by processing one key at a time.
 * This class handles cache warming and performance measurement.
 */
public class SingleKeyRunnableThd implements Runnable {
    
    // Thread management
    private Thread thread;
    private final String threadName;
    
    // Cache and workload
    private final ISimpleCache cache;
    private final List<String> workloadKeys;
    private final int benchmarkWorkloadSize;
    
    // Performance tracking
    private int localHitCount = 0;
    private final List<Boolean> hitMissRecord = new ArrayList<>();
    private final List<Integer> hitCountDistribution = new ArrayList<>();
    
    // Configuration constants
    private static final int EVALUATION_TABLE_SIZE = 26;
    private final int groupSize = EVALUATION_TABLE_SIZE;
    
    /**
     * Creates a new benchmark thread.
     * 
     * @param name Thread name for identification
     * @param cache Cache instance to benchmark
     * @param workloadKeys List of keys to use in benchmark
     * @param benchmarkWorkloadSize Size of the benchmark workload
     */
    public SingleKeyRunnableThd(String name, ISimpleCache cache, 
                               List<String> workloadKeys, int benchmarkWorkloadSize) {
        this.threadName = name;
        this.cache = cache;
        this.workloadKeys = workloadKeys;
        this.benchmarkWorkloadSize = benchmarkWorkloadSize;
        
        // Initialize hit count distribution tracking
        for (int i = 0; i <= groupSize; i++) {
            hitCountDistribution.add(0);
        }
        
        System.out.println("Created benchmark thread: " + this.threadName);
    }
    
    /**
     * Returns the number of cache hits recorded by this thread.
     */
    public int getLocalHitCount() {
        return localHitCount;
    }
    
    /**
     * Returns the detailed hit/miss record for each request.
     */
    public List<Boolean> getHitMissRecord() {
        return hitMissRecord;
    }
    
    /**
     * Warms up the cache using sequential access pattern.
     * 
     * @param warmUpPercentage Percentage of workload to use for warm-up (0-100)
     */
    public void warmUpCacheSequentially(int warmUpPercentage) {
        System.out.println("Starting sequential warm-up with " + warmUpPercentage + "% of workload");
        
        int warmUpSize = benchmarkWorkloadSize * warmUpPercentage / 100 * EVALUATION_TABLE_SIZE;
        
        for (int i = 0; i < warmUpSize && i < workloadKeys.size(); i++) {
            cache.request(workloadKeys.get(i));
        }
        
        System.out.println("Sequential warm-up completed");
    }
    
    /**
     * Warms up the cache using random access pattern.
     * 
     * @param multiplier Multiple of cache size to use for warm-up
     */
    public void warmUpCacheRandomly(int multiplier) {
        int warmUpSize = cache.getCacheSize() * multiplier;
        
        System.out.println("Starting random warm-up with " + multiplier + "x cache size");
        System.out.println("Warm-up size: " + warmUpSize + ", Available keys: " + workloadKeys.size());
        
        if (warmUpSize >= workloadKeys.size()) {
            // Use sequential access if warm-up size exceeds available keys
            System.out.println("Using sequential access (warm-up size >= available keys)");
            for (int i = 0; i < workloadKeys.size(); i++) {
                cache.request(workloadKeys.get(i));
            }
        } else {
            // Use random access
            Random random = new Random(warmUpSize); // Use warm-up size as seed for reproducibility
            int maxIndex = workloadKeys.size() - 1;
            
            for (int i = 0; i < warmUpSize; i++) {
                int randomIndex = random.nextInt(maxIndex);
                cache.request(workloadKeys.get(randomIndex));
            }
        }
        
        System.out.println("Random warm-up completed");
    }
    
    /**
     * Outputs the cumulative distribution function of hit counts.
     */
    public void printHitCountCDF() {
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
     * Main benchmark execution logic.
     * Processes the benchmark workload and records performance metrics.
     */
    @Override
    public void run() {
        int totalRequests = 0;
        int currentGroupHits = 0;
        
        // Calculate the starting index for benchmark portion of workload
        int startIndex = workloadKeys.size() - (benchmarkWorkloadSize * EVALUATION_TABLE_SIZE);
        
        // Process each key in the benchmark workload
        for (int i = startIndex; i < workloadKeys.size(); i++) {
            totalRequests++;
            
            // Make cache request and record result
            boolean isHit = cache.request(workloadKeys.get(i));
            hitMissRecord.add(isHit);
            
            if (isHit) {
                localHitCount++;
                currentGroupHits++;
            }
            
            // Record hit count distribution every groupSize requests
            if (totalRequests % groupSize == 0) {
                int currentCount = hitCountDistribution.get(currentGroupHits);
                hitCountDistribution.set(currentGroupHits, currentCount + 1);
                currentGroupHits = 0;
            }
        }
        
        System.out.println("Thread " + threadName + " completed. Total hits: " + localHitCount);
    }
    
    /**
     * Starts the benchmark thread execution.
     */
    public void start() {
        System.out.println("Starting thread: " + threadName);
        
        if (thread == null) {
            thread = new Thread(this, threadName);
            thread.start();
        }
    }
}
