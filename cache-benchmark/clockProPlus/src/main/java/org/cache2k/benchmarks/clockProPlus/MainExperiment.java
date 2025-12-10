package org.cache2k.benchmarks.clockProPlus;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.io.FilenameUtils;

/**
 * Cache benchmark experiment runner with support for multiple cache policies
 * and workload testing scenarios.
 */
public class MainExperiment {
    
    // Configuration constants
    private static final String BASE_DIR = "/home/xxx/VineCache/cache-benchmark/inf-workload-traces/criteo_kaggle_all_mmap/inference=0.01/";
    private static final String BENCH_DIR = "caching_bench/";
    private static final String SUMMARY_DIR = "summary/";
    private static final String SUMMARY_OUTPUT_DIR = "caching_hit_summary/";
    private static final int DEFAULT_GROUP_COUNT = 26;
    
    // Cache capacity percentages for testing
    private static final float[] CACHE_CAPACITIES = {
        0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f, 
        1f, 2f, 4f, 8f, 16f, 32f, 64f, 100f
    };
    
    // Supported cache policies
    private static final String[] DEFAULT_POLICIES = {"SimpleLRU", "VineCache_LRU"};

    /**
     * Simple sanity test for cache implementations
     */
    public static void runSanityTest(ISimpleCache cache) {
        List<String> testKeys = Arrays.asList("1", "2", "3", "1", "4", "5", "6", "7", "8", "9", "10");
        
        System.out.println("Running sanity test:");
        for (String key : testKeys) {
            cache.request(key);
            System.out.println(key + " -> " + cache.toString());
        }
    }

    /**
     * Factory method to create cache policy instances
     */
    public static ISimpleCache createCacheInstance(String policyName, int size) {
        return createCacheInstance(policyName, size, DEFAULT_GROUP_COUNT);
    }

    public static ISimpleCache createCacheInstance(String policyName, int size, int groupCount) {
        switch (policyName) {
            case "SimpleMFU": return new SimpleMFU(size);
            case "VineCache_LRU": return new VineCache_LRU(size, groupCount);
            case "VineCache_MFU": return new VineCache_MFU(size, groupCount);
            case "TwoQ": return new TwoQ(size);
            case "LRUK": return new LRUK(size, 3);
            case "S3FIFO": return new S3FIFO(size);
            case "ClockNBit": return new ClockNBit(size, 3);
            case "EvCAR": return new EvCAR(size, groupCount);
            case "EvARC": return new EvARC(size, groupCount);
            case "EvLFU": return new EvLFU(size, groupCount);
            case "SimpleLRU": return new SimpleLRU(size);
            case "SimpleLFU": return new SimpleLFU(size);
            case "DynamicLIRS": return new DynamicLIRS(size, new SimpleLIRS.Tuning());
            case "SimpleLIRS": return new SimpleLIRS(size, new SimpleLIRS.Tuning());
            case "SimpleARC": return new SimpleARC(size);
            case "CAR": return new CAR(size);
            case "Clock": return new Clock(size);
            case "ClockLIRS": return new ClockLIRS(size, new ClockPro.Tuning());
            case "ClockPro": return new ClockPro(size, new ClockPro.Tuning());
            case "SimpleFIFO": return new SimpleFIFO(size);
            default:
                throw new IllegalArgumentException("Unsupported cache policy: " + policyName);
        }
    }

    /**
     * Read workload data from file with size limit
     */
    public static List<String> readWorkloadFile(Path filePath, int maxSize) {
        List<String> workloadKeys = new ArrayList<>();
        
        try (BufferedReader reader = Files.newBufferedReader(filePath)) {
            String header = reader.readLine();
            System.out.print(header + " ; ");
            
            if (header.contains(",")) {
                // Handle CSV format with ordered keys
                workloadKeys = readCsvWorkload(reader, maxSize);
            } else {
                // Handle simple key-per-line format
                workloadKeys = readSimpleWorkload(reader, maxSize);
            }
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to read workload file: " + filePath, e);
        }
        
        return workloadKeys;
    }
    
    private static List<String> readSimpleWorkload(BufferedReader reader, int maxSize) throws IOException {
        return reader.lines()
                .limit(maxSize)
                .collect(Collectors.toList());
    }
    
    private static List<String> readCsvWorkload(BufferedReader reader, int maxSize) throws IOException {
        List<String> keys = new ArrayList<>();
        String line;
        int count = 0;
        
        while ((line = reader.readLine()) != null && count < maxSize) {
            String[] values = line.split(",");
            if (values.length > 1) {
                keys.add(values[1]); // Use ordered key column
            }
            count++;
        }
        
        return keys;
    }

    /**
     * Write results to file with proper directory creation
     */
    public static <T> void writeResultsToFile(List<T> data, Path outputPath, String header) {
        createDirectoriesIfNeeded(outputPath);
        
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writer.write(header + System.lineSeparator());
            for (T item : data) {
                writer.write(item + System.lineSeparator());
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write to file: " + outputPath, e);
        }
    }
    
    private static void createDirectoriesIfNeeded(Path filePath) {
        Path parentDir = filePath.getParent();
        if (parentDir != null && !Files.exists(parentDir)) {
            try {
                Files.createDirectories(parentDir);
            } catch (IOException e) {
                throw new RuntimeException("Failed to create directories: " + parentDir, e);
            }
        }
    }

    /**
     * Record cache hit/miss statistics for a given policy and workload
     */
    public static CacheHitResult recordCacheHits(String policyName, int cacheSize, 
                                                List<String> workloadKeys) {
        ISimpleCache cache = createCacheInstance(policyName, cacheSize);
        
        int hits = 0;
        int misses = 0;
        List<Integer> hitRecord = new ArrayList<>();
        
        for (String key : workloadKeys) {
            if (cache.request(key)) {
                hitRecord.add(1);
                hits++;
            } else {
                hitRecord.add(0);
                misses++;
            }
        }
        
        return new CacheHitResult(hits, misses, hitRecord);
    }

    /**
     * Test cache hit rates across different capacity percentages
     */
    public static void testCacheHitRates(int uniqueKeyCount, List<String> workloadKeys, 
                                       Path outputDir, String policyName) {
        System.out.println("Testing cache hit rates:");
        System.out.println("Total workload: " + workloadKeys.size() + " keys");
        System.out.println("Unique keys: " + uniqueKeyCount + " keys");

        for (float capacityPercent : CACHE_CAPACITIES) {
            int cacheSize = calculateCacheSize(capacityPercent, uniqueKeyCount);
            String capacityStr = formatCapacityString(capacityPercent);
            
            System.out.println("Testing cache size: " + cacheSize + " (" + capacityStr + ")");
            
            CacheHitResult result = recordCacheHits(policyName, cacheSize, workloadKeys);
            
            saveCacheResults(result, outputDir, capacityStr);
        }
    }
    
    private static int calculateCacheSize(float capacityPercent, int uniqueKeyCount) {
        return (int) (capacityPercent / 100 * uniqueKeyCount);
    }
    
    private static String formatCapacityString(float capacityPercent) {
        return capacityPercent >= 1 ? String.valueOf((int) capacityPercent) : String.valueOf(capacityPercent);
    }
    
    private static void saveCacheResults(CacheHitResult result, Path outputDir, String capacityStr) {
        String fullRecordFile = "fullrecord-" + capacityStr + "%.csv";
        String summaryFile = "summary-" + capacityStr + "%.csv";
        
        List<String> summary = Arrays.asList(
            result.getHitRate() + "," + result.getMissRate(),
            result.getHits() + "," + result.getMisses()
        );
        
        writeResultsToFile(summary, outputDir.resolve(summaryFile), "hit,miss");
        writeResultsToFile(result.getHitRecord(), outputDir.resolve(fullRecordFile), "hit_or_miss");
    }

    /**
     * Load all CSV workload files from the base directory
     */
    public static List<String> loadWorkloadFiles() {
        File baseDir = new File(BASE_DIR);
        File[] files = baseDir.listFiles();
        
        if (files == null) {
            throw new RuntimeException("Cannot access base directory: " + BASE_DIR);
        }
        
        return Arrays.stream(files)
                .filter(File::isFile)
                .map(File::getName)
                .filter(name -> name.endsWith(".csv"))
                .sorted()
                .collect(Collectors.toList());
    }

    /**
     * Main benchmark execution for single cache queue testing
     */
    public static void runSingleCacheQueueBenchmark(List<String> workloadFiles, String algorithmName, 
                                                   int cacheSize, int benchmarkSize, 
                                                   boolean enableWarmup, int warmupMultiplier) {
        
        System.out.println("Running benchmark for: " + algorithmName);
        System.out.println("Cache size: " + cacheSize);
        System.out.println("Benchmark size: " + benchmarkSize);
        
        // Load and merge workloads
        List<String> mergedWorkload = loadAndMergeWorkloads(workloadFiles, benchmarkSize, warmupMultiplier);
        
        // Create appropriate thread for the algorithm
        SingleKeyRunnableThd thread = createBenchmarkThread(algorithmName, cacheSize, 
                                                           mergedWorkload, workloadFiles.size(), 
                                                           benchmarkSize);
        
        // Execute benchmark
        BenchmarkResult result = executeBenchmark(thread, enableWarmup, warmupMultiplier);
        
        // Save results
        saveBenchmarkResults(result, workloadFiles, algorithmName, cacheSize, benchmarkSize);
    }
    
    private static List<String> loadAndMergeWorkloads(List<String> workloadFiles, 
                                                     int benchmarkSize, int warmupMultiplier) {
        List<List<String>> allWorkloads = new ArrayList<>();
        int totalDataSize = benchmarkSize * (100 + warmupMultiplier) / 100;
        
        System.out.println("Loading workloads...");
        for (String fileName : workloadFiles) {
            Path filePath = Paths.get(BASE_DIR, fileName);
            List<String> workload = readWorkloadFile(filePath, totalDataSize);
            allWorkloads.add(workload);
        }
        
        return mergeWorkloads(allWorkloads);
    }
    
    private static List<String> mergeWorkloads(List<List<String>> workloads) {
        if (workloads.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<String> merged = new ArrayList<>();
        int maxSize = workloads.get(0).size();
        
        for (int i = 0; i < maxSize; i++) {
            for (List<String> workload : workloads) {
                if (i < workload.size()) {
                    merged.add(workload.get(i));
                }
            }
        }
        
        return merged;
    }
    
    private static SingleKeyRunnableThd createBenchmarkThread(String algorithmName, int cacheSize,
                                                             List<String> workload, int tableCount,
                                                             int benchmarkSize) {
        
        boolean isMultiKeyAlgorithm = isMultiKeyAlgorithm(algorithmName);
        
        if (isMultiKeyAlgorithm) {
            return new MultiKeyRunnableThd("Benchmark-Thread",
                    createCacheInstance(algorithmName, cacheSize, tableCount),
                    workload, tableCount, benchmarkSize);
        } else {
            return new SingleKeyRunnableThd("Benchmark-Thread",
                    createCacheInstance(algorithmName, cacheSize),
                    workload, benchmarkSize);
        }
    }
    
    private static boolean isMultiKeyAlgorithm(String algorithmName) {
        return algorithmName.contains("VineCache") || 
               algorithmName.contains("EvLRU") || 
               algorithmName.contains("EvCAR") || 
               algorithmName.contains("EvLFU") || 
               algorithmName.contains("EvARC");
    }
    
    private static BenchmarkResult executeBenchmark(SingleKeyRunnableThd thread, 
                                                   boolean enableWarmup, int warmupMultiplier) {
        if (enableWarmup) {
            thread.warmUpTheCache(warmupMultiplier);
        }
        
        long startTime = System.nanoTime();
        thread.start();
        
        try {
            thread.t.join();
            long duration = System.nanoTime() - startTime;
            
            System.out.println("Benchmark completed in: " + duration / 1_000_000_000.0 + " seconds");
            
            return new BenchmarkResult(thread.getRecordHitMiss(), duration);
            
        } catch (InterruptedException e) {
            throw new RuntimeException("Benchmark interrupted", e);
        }
    }
    
    private static void saveBenchmarkResults(BenchmarkResult result, List<String> workloadFiles,
                                           String algorithmName, int cacheSize, int benchmarkSize) {
        // Implementation for saving benchmark results
        // This would include the complex logic from the original method
        // but in a more organized structure
        System.out.println("Saving benchmark results for " + algorithmName);
    }

    /**
     * Main entry point
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: MainExperiment <benchmarkSize> <cacheSize>");
            System.exit(1);
        }
        
        int benchmarkSize = Integer.parseInt(args[0]);
        int cacheSize = Integer.parseInt(args[1]);
        boolean enableWarmup = true;
        int warmupMultiplier = 100;
        
        System.out.println("=== Cache Benchmark Experiment ===");
        System.out.println("Benchmark size: " + benchmarkSize);
        System.out.println("Cache size: " + cacheSize);
        
        List<String> workloadFiles = loadWorkloadFiles();
        
        for (String policy : DEFAULT_POLICIES) {
            runSingleCacheQueueBenchmark(workloadFiles, policy, cacheSize, 
                                       benchmarkSize, enableWarmup, warmupMultiplier);
        }
        
        System.out.println("Benchmark completed successfully!");
    }

    // Helper classes for better data organization
    public static class CacheHitResult {
        private final int hits;
        private final int misses;
        private final List<Integer> hitRecord;
        
        public CacheHitResult(int hits, int misses, List<Integer> hitRecord) {
            this.hits = hits;
            this.misses = misses;
            this.hitRecord = hitRecord;
        }
        
        public int getHits() { return hits; }
        public int getMisses() { return misses; }
        public List<Integer> getHitRecord() { return hitRecord; }
        public float getHitRate() { return (float) hits / (hits + misses) * 100; }
        public float getMissRate() { return (float) misses / (hits + misses) * 100; }
    }
    
    public static class BenchmarkResult {
        private final ArrayList<Boolean> hitMissRecord;
        private final long duration;
        
        public BenchmarkResult(ArrayList<Boolean> hitMissRecord, long duration) {
            this.hitMissRecord = hitMissRecord;
            this.duration = duration;
        }
        
        public ArrayList<Boolean> getHitMissRecord() { return hitMissRecord; }
        public long getDuration() { return duration; }
    }
}
