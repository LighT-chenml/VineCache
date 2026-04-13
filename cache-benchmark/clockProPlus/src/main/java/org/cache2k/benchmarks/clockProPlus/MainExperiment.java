package org.cache2k.benchmarks.clockProPlus;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.io.FilenameUtils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Cache benchmark experiment runner with support for multiple cache policies
 * and workload testing scenarios.
 */
public class MainExperiment {
    
    // --- Configuration Constants ---
    
    /** Base directory for workload traces */
    private static final String BASE_WORKLOAD_DIR = System.getProperty("workload.dir", 
        "/home/xxx/VineCache/cache-benchmark/inf-workload-traces/criteo_kaggle_all_mmap/inference=0.01/");
    
    /** Output directory names */
    private static final String DIR_BENCHMARK = "caching_bench/";
    private static final String DIR_SUMMARY = "summary/";
    private static final String DIR_HIT_SUMMARY = "caching_hit_summary/";
    
    /** Default configuration values */
    private static final int DEFAULT_GROUP_COUNT = 26;
    private static final int DEFAULT_WARMUP_MULTIPLIER = 100;
    
    /** Cache capacity percentages to test */
    private static final float[] CACHE_CAPACITY_POINTS = {
        0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f, 
        1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f, 64.00f, 100.00f
    };
    
    /** Default policies to run in main */
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
            case "CAR":           return new CAR(size);
            case "Clock":         return new Clock(size);
            case "ClockLIRS":     return new ClockLIRS(size, new ClockPro.Tuning());
            case "ClockNBit":     return new ClockNBit(size, 3);
            case "ClockPro":      return new ClockPro(size, new ClockPro.Tuning());
            case "DynamicLIRS":   return new DynamicLIRS(size, new SimpleLIRS.Tuning());
            case "EvARC":         return new EvARC(size, groupCount);
            case "EvCAR":         return new EvCAR(size, groupCount);
            case "EvLFU":         return new EvLFU(size, groupCount);
            case "LRUK":          return new LRUK(size, 3);
            case "S3FIFO":        return new S3FIFO(size);
            case "SimpleARC":     return new SimpleARC(size);
            case "SimpleFIFO":    return new SimpleFIFO(size);
            case "SimpleLFU":     return new SimpleLFU(size);
            case "SimpleLIRS":    return new SimpleLIRS(size, new SimpleLIRS.Tuning());
            case "SimpleLRU":     return new SimpleLRU(size);
            case "SimpleMFU":     return new SimpleMFU(size);
            case "TwoQ":          return new TwoQ(size);
            case "VineCache_LRU": return new VineCache_LRU(size, groupCount);
            case "VineCache_MFU": return new VineCache_MFU(size, groupCount);
            default:
                throw new IllegalArgumentException("Unsupported cache policy: " + policyName);
        }
    }

    /**
     * Read workload data from file with size limit
     */
    public static List<String> readWorkload(Path filePath, int maxSize) {
        try (BufferedReader reader = Files.newBufferedReader(filePath)) {
            String firstLine = reader.readLine();
            if (firstLine == null) return Collections.emptyList();
            
            System.out.print(firstLine + " ; ");
            
            // CSV format typically contains commas, handle accordingly
            if (firstLine.contains(",")) {
                return readCsvFormat(reader, maxSize);
            } else {
                return readListFormat(reader, maxSize, firstLine);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Error reading workload: " + filePath, e);
        }
    }

    private static List<String> readListFormat(BufferedReader reader, int maxSize, String firstLine) throws IOException {
        List<String> keys = new ArrayList<>();
        keys.add(firstLine.trim());
        
        String line;
        while ((line = reader.readLine()) != null && keys.size() < maxSize) {
            keys.add(line.trim());
        }
        return keys;
    }

    private static List<String> readCsvFormat(BufferedReader reader, int maxSize) throws IOException {
        List<String> keys = new ArrayList<>();
        String line;
        
        while ((line = reader.readLine()) != null && keys.size() < maxSize) {
            String[] parts = line.split(",");
            if (parts.length > 1) {
                keys.add(parts[1].trim()); // Column 1 is usually the ordered key
            }
        }
        return keys;
    }

    /**
     * Write results to file with proper directory creation
     */
    public static <T> void writeDataToFile(List<T> data, Path path, String header) {
        ensureDirectoryExists(path);
        
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            if (header != null && !header.isEmpty()) {
                writer.write(header + System.lineSeparator());
            }
            for (T item : data) {
                writer.write(item.toString() + System.lineSeparator());
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Error writing to file: " + path, e);
        }
    }

    private static void ensureDirectoryExists(Path filePath) {
        Path parent = filePath.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                throw new UncheckedIOException("Could not create directory: " + parent, e);
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

        for (float capacityPercent : CACHE_CAPACITY_POINTS) {
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
        
        writeDataToFile(summary, outputDir.resolve(summaryFile), "hit,miss");
        writeDataToFile(result.getHitRecord(), outputDir.resolve(fullRecordFile), "hit_or_miss");
    }

    /**
     * Load all CSV workload files from the base directory
     */
    public static List<String> loadWorkloadFiles() {
        File baseDir = new File(BASE_WORKLOAD_DIR);
        File[] files = baseDir.listFiles();
        
        if (files == null) {
            throw new RuntimeException("Cannot access base directory: " + BASE_WORKLOAD_DIR);
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
            Path filePath = Paths.get(BASE_WORKLOAD_DIR, fileName);
            List<String> workload = readWorkload(filePath, totalDataSize);
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
            thread.join();
            long duration = System.nanoTime() - startTime;
            
            System.out.printf("Benchmark completed in %.3f seconds%n", duration / 1_000_000_000.0);
            
            return new BenchmarkResult(thread.getRecordHitMiss(), duration);
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Benchmark interrupted", e);
        }
    }
    
    private static void saveBenchmarkResults(BenchmarkResult result, List<String> workloadFiles,
                                           String algorithmName, int cacheSize, int benchmarkSize) {
        
        Path outputDir = Paths.get(DIR_SUMMARY, algorithmName, String.valueOf(cacheSize));
        String timestamp = String.valueOf(System.currentTimeMillis());
        
        int hits = 0;
        int total = result.getHitMissRecord().size();
        for (Boolean hit : result.getHitMissRecord()) {
            if (hit) hits++;
        }
        
        double hitRate = (double) hits / total;
        
        System.out.printf("Results for %s: Hits=%d, Total=%d, HitRate=%.4f, Time=%dms%n", 
            algorithmName, hits, total, hitRate, result.getDuration() / 1_000_000);
        
        List<String> summary = Arrays.asList(
            "Algorithm," + algorithmName,
            "CacheSize," + cacheSize,
            "BenchmarkSize," + benchmarkSize,
            "Hits," + hits,
            "Total," + total,
            "HitRate," + hitRate,
            "DurationMs," + (result.getDuration() / 1_000_000)
        );
        
        writeDataToFile(summary, outputDir.resolve("experiment_summary_" + timestamp + ".csv"), "Key,Value");
        
        List<Integer> bitRecord = result.getHitMissRecord().stream()
            .map(b -> b ? 1 : 0)
            .collect(Collectors.toList());
            
        writeDataToFile(bitRecord, outputDir.resolve("hit_record_" + timestamp + ".csv"), "hit_or_miss");
    }

    /**
     * Main entry point
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: MainExperiment <benchmarkSize> <cacheSize> [warmupEnable]");
            System.out.println("Using default values: size=100000, cache=1000");
            runDefaultExperiment(100000, 1000);
            return;
        }
        
        int benchmarkSize = Integer.parseInt(args[0]);
        int cacheSize = Integer.parseInt(args[1]);
        boolean enableWarmup = args.length <= 2 || Boolean.parseBoolean(args[2]);
        
        runExperiment(benchmarkSize, cacheSize, enableWarmup);
    }

    private static void runDefaultExperiment(int benchmarkSize, int cacheSize) {
        runExperiment(benchmarkSize, cacheSize, true);
    }

    private static void runExperiment(int benchmarkSize, int cacheSize, boolean enableWarmup) {
        System.out.println("=== Starting Cache Benchmark Experiment ===");
        System.out.println("Configuration:");
        System.out.println("  Benchmark Size: " + benchmarkSize);
        System.out.println("  Cache Size:     " + cacheSize);
        System.out.println("  Warmup Enabled: " + enableWarmup);
        System.out.println("-------------------------------------------");
        
        List<String> workloadFiles = loadWorkloadFiles();
        if (workloadFiles.isEmpty()) {
            System.err.println("No workload files found in " + BASE_WORKLOAD_DIR);
            return;
        }
        
        for (String policy : DEFAULT_POLICIES) {
            try {
                runSingleCacheQueueBenchmark(workloadFiles, policy, cacheSize, 
                                           benchmarkSize, enableWarmup, DEFAULT_WARMUP_MULTIPLIER);
            } catch (Exception e) {
                System.err.println("Error running benchmark for " + policy + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("-------------------------------------------");
        System.out.println("Experiment cycle completed successfully.");
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
