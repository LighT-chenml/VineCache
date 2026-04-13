package org.cache2k.benchmarks.clockProPlus;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A thread that executes cache benchmark by processing one key at a time.
 * This class handles cache warming and performance measurement.
 */
public class SingleKeyRunnableThd implements Runnable {
    
    // Constants
    private static final int ITEMS_PER_GROUP = 26; // Default grouping for hit distribution

    // Execution State
    protected Thread executorThread;
    private final String threadName;

    // Cache and Workload Details
    protected final ISimpleCache cache;
    protected final List<String> workloadKeys;
    protected final int benchmarkWorkloadSize;

    // Statistics and Collections
    protected int localHitCount = 0;
    protected final ArrayList<Boolean> recordHitMiss = new ArrayList<>();
    protected final List<Integer> hitCountDistribution = new ArrayList<>();
    

    
    /**
     * Initializes a benchmark thread with the specified cache and workload.
     *
     * @param name                  The name of the thread.
     * @param cache                 The cache implementation to benchmark.
     * @param workloadKeys          The complete sequence of keys for the benchmark.
     * @param benchmarkWorkloadSize The number of operations to perform in the evaluation phase.
     */
    public SingleKeyRunnableThd(String name, ISimpleCache cache, 
                               List<String> workloadKeys, int benchmarkWorkloadSize) {
        this.threadName = name;
        this.cache = cache;
        this.workloadKeys = workloadKeys;
        this.benchmarkWorkloadSize = benchmarkWorkloadSize;

        initializeDistributionTracking();
        System.out.println("[Benchmark] Created thread: " + name);
    }

    private void initializeDistributionTracking() {
        hitCountDistribution.clear();
        for (int i = 0; i <= ITEMS_PER_GROUP; i++) {
            hitCountDistribution.add(0);
        }
    }
    
    /**
     * Returns the number of cache hits recorded by this thread.
     */
    public ArrayList<Boolean> getRecordHitMiss() {
        return recordHitMiss;
    }
    
    /**
     * Performs a sequential warm-up of the cache using a portion of the workload.
     *
     * @param warmUpPercentage Percentage of benchmark workload to use (0-100).
     */
    public void warmUpTheCache(int warmUpPercentage) {
        int warmUpCount = (benchmarkWorkloadSize * warmUpPercentage / 100) * ITEMS_PER_GROUP;
        System.out.println("[Warm-up] Sequential start (" + warmUpPercentage + "%, size=" + warmUpCount + ")");

        for (int i = 0; i < warmUpCount && i < workloadKeys.size(); i++) {
            cache.request(workloadKeys.get(i));
        }
        System.out.println("[Warm-up] Sequential completed.");
    }
    
    /**
     * Performs a random warm-up of the cache using keys from the workload.
     *
     * @param multiplier Number of times the cache size to request.
     */
    public void warmUpCacheRandomly(int multiplier) {
        int warmUpCount = cache.getCacheSize() * multiplier;
        System.out.println("[Warm-up] Random start (" + multiplier + "x cache size, count=" + warmUpCount + ")");

        if (warmUpCount >= workloadKeys.size()) {
            System.out.println("[Warm-up] Switching to sequential (count exceeds available keys).");
            workloadKeys.forEach(cache::request);
        } else {
            Random random = new Random(warmUpCount); 
            int bound = workloadKeys.size();
            for (int i = 0; i < warmUpCount; i++) {
                cache.request(workloadKeys.get(random.nextInt(bound)));
            }
        }
        System.out.println("[Warm-up] Random completed.");
    }
    
    /**
     * Records and prints the Cumulative Distribution Function (CDF) of hit counts.
     */
    public void printHitCountCDF() {
        int totalGroups = hitCountDistribution.stream().mapToInt(Integer::intValue).sum();
        if (totalGroups == 0) return;

        System.out.println("[Stats] Hit Count CDF:");
        double cumulativeProb = 0.0;
        for (int i = ITEMS_PER_GROUP; i >= 0; i--) {
            cumulativeProb += (double) hitCountDistribution.get(i) / totalGroups;
            System.out.printf("  P(hits >= %d) = %.4f%n", i, cumulativeProb);
        }
    }

    @Override
    public void run() {
        int totalProcessed = 0;
        int groupHits = 0;
        int startIndex = Math.max(0, workloadKeys.size() - (benchmarkWorkloadSize * ITEMS_PER_GROUP));

        for (int i = startIndex; i < workloadKeys.size(); i++) {
            totalProcessed++;
            boolean isHit = cache.request(workloadKeys.get(i));
            recordHitMiss.add(isHit);

            if (isHit) {
                localHitCount++;
                groupHits++;
            }

            // Update distribution every group interval
            if (totalProcessed % ITEMS_PER_GROUP == 0) {
                hitCountDistribution.set(groupHits, hitCountDistribution.get(groupHits) + 1);
                groupHits = 0;
            }
        }
        System.out.println("[Execution] Thread " + threadName + " completed. Total Hits: " + localHitCount);
    }
    
    public void join() throws InterruptedException {
        if (executorThread != null) {
            executorThread.join();
        }
    }

    /**
     * Starts the asynchronous execution of the benchmark.
     */
    public void start() {
        System.out.println("[Execution] Starting thread: " + threadName);
        if (executorThread == null) {
            executorThread = new Thread(this, threadName);
            executorThread.start();
        }
    }
}
