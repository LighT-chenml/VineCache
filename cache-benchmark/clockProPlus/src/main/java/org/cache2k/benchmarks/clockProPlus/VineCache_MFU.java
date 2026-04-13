package org.cache2k.benchmarks.clockProPlus;

import java.util.*;

/**
 * A multi-level cache implementation with MFU (Most Frequently Used) eviction 
 * and reinforcement learning (Q-learning) based parameter tuning.
 *
 * <p>Key features:
 * <ul>
 *   <li>Hierarchical cache levels based on "experience" values.</li>
 *   <li>MFU eviction policy within each cache level.</li>
 *   <li>Automatic parameter tuning using a Q-learning agent.</li>
 *   <li>Native support for multi-key (group) requests.</li>
 * </ul>
 */
public class VineCache_MFU implements IMultiKeyCache {
    
    // --- Constants ---
    private static final int DEFAULT_NUM_LEVELS = 30;
    private static final double[] PRIORITY_BOUNDARIES = {0.4, 0.8, 0.95};

    // --- Configuration ---
    private final int capacity;
    private final int groupSize;
    private final int numLevels;
    private final int[] hitThresholds;
    private final int[] levelBoundaries;

    // --- Core Data Structures ---
    private final List<SimpleMFU> mfuLevels;
    private final Map<String, Double> experienceMap;
    private final Map<String, Integer> lastAccessMap;
    private final Random random;

    // --- Adaptive Components ---
    private final CacheConfig config;
    private final QLearningAgent adaptiveAgent;

    // --- Operational Statistics ---
    private int globalClock;
    private int recentEvictions;
    private int groupHitCount;
    private int aggregateLevelSum;
    
    // ==================== Constructor ====================
    public VineCache_MFU(int capacity, int groupSize) {
        this.capacity = capacity;
        this.groupSize = groupSize;
        this.numLevels = DEFAULT_NUM_LEVELS;
        this.random = new Random(42); // Standard seed for reproducibility

        // Initialize thresholds for hit counts
        this.hitThresholds = new int[PRIORITY_BOUNDARIES.length];
        for (int i = 0; i < PRIORITY_BOUNDARIES.length; i++) {
            this.hitThresholds[i] = (int) (PRIORITY_BOUNDARIES[i] * groupSize);
        }

        // Initialize boundaries for experience levels (specific to MFU)
        this.levelBoundaries = new int[numLevels - 1];
        levelBoundaries[0] = 8;
        for (int i = 1; i < numLevels - 1; i++) {
            levelBoundaries[i] = levelBoundaries[i - 1] + levelBoundaries[0] * (1 << i);
        }

        // Initialize cache hierarchies
        this.mfuLevels = new ArrayList<>(numLevels);
        for (int i = 0; i < numLevels; i++) {
            mfuLevels.add(new SimpleMFU(capacity));
        }

        this.experienceMap = new HashMap<>();
        this.lastAccessMap = new HashMap<>();
        this.config = new CacheConfig(capacity);
        this.adaptiveAgent = new QLearningAgent();

        this.globalClock = 0;
        this.recentEvictions = 0;
        this.groupHitCount = 0;
        this.aggregateLevelSum = 0;
    }
    
    // --- Request Processing ---

    @Override
    public List<Boolean> request(List<String> groupKeys) {
        globalClock++;
        
        GroupRequestResult result = evaluateGroupRequest(groupKeys);
        applyExperienceUpdates(groupKeys, result);
        triggerAdaptiveCycle();
        
        return result.hitMissStatus;
    }

    private GroupRequestResult evaluateGroupRequest(List<String> groupKeys) {
        List<Boolean> status = new ArrayList<>(groupKeys.size());
        List<Integer> levels = new ArrayList<>(groupKeys.size());
        int totalHits = 0;

        for (String key : groupKeys) {
            boolean isHit = experienceMap.containsKey(key);
            status.add(isHit);
            if (isHit) {
                totalHits++;
                levels.add(determineLevel(experienceMap.get(key)));
            } else {
                levels.add(0);
            }
        }

        if (totalHits == groupSize) {
            groupHitCount++;
        }

        return new GroupRequestResult(status, levels, totalHits);
    }

    private void applyExperienceUpdates(List<String> groupKeys, GroupRequestResult result) {
        double rewardBase = calculateRewardBase(result.hits);
        List<Double> deltas = computeExperienceDeltas(result.levels, rewardBase);

        for (int i = 0; i < groupKeys.size(); i++) {
            recordAccess(groupKeys.get(i), deltas.get(i), result.hitMissStatus.get(i));
        }
    }
    
    // ==================== Experience Calculation ====================
    // --- Experience & Level Calculations ---

    private double calculateRewardBase(int hitCount) {
        double rawReward = (config.experienceFunction == 0) 
                ? calculateLinearReward(hitCount) 
                : Math.pow(Math.E, (hitCount - 10) / 2.0);
        
        return rawReward * config.ageMultiplier / 100.0;
    }

    private double calculateLinearReward(int hits) {
        if (hits < hitThresholds[0]) return 0;
        if (hits < hitThresholds[1]) return hits - hitThresholds[0];
        
        if (hits < hitThresholds[2]) {
            return (hitThresholds[1] - hitThresholds[0]) + (hits - hitThresholds[1]) * 2.0;
        }
        
        return (hitThresholds[1] - hitThresholds[0]) + 
               (hitThresholds[2] - hitThresholds[1]) * 2.0 + 
               (hits - hitThresholds[2]) * 8.0;
    }

    private List<Double> computeExperienceDeltas(List<Integer> levels, double rewardBase) {
        List<Double> deltas = new ArrayList<>(levels.size());
        for (int level : levels) {
            double mmrFactor = calculateMMRFactor(levels, level);
            deltas.add(rewardBase * mmrFactor);
        }
        return deltas;
    }

    private double calculateMMRFactor(List<Integer> levels, int currentLevel) {
        int cubicSum = 0;
        for (int level : levels) {
            int diff = level - currentLevel;
            cubicSum += (diff * diff * diff);
        }

        double score = 0;
        if (cubicSum > 0) score = Math.log(cubicSum);
        else if (cubicSum < 0) score = -Math.log(-cubicSum);

        return sigmoid(score * config.mmrRate);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private int determineLevel(double experience) {
        for (int i = 0; i < numLevels - 1; i++) {
            if (experience <= levelBoundaries[i]) return i;
        }
        return numLevels - 1;
    }
    
    private void recordAccess(String key, double delta, boolean isHit) {
        if (isHit) {
            updateExistingEntry(key, delta);
        } else {
            createNewEntry(key, delta);
        }
        lastAccessMap.put(key, globalClock);
        handleAging();
    }

    private void updateExistingEntry(String key, double delta) {
        if (!experienceMap.containsKey(key)) return;

        double oldExp = experienceMap.get(key);
        double newExp = oldExp + delta;
        
        int oldLevel = determineLevel(oldExp);
        int newLevel = determineLevel(newExp);

        if (oldLevel != newLevel) {
            int currentFreq = mfuLevels.get(oldLevel).remove(key);
            mfuLevels.get(newLevel).forceInsert(key, currentFreq + 1);
            aggregateLevelSum += (newLevel - oldLevel);
        } else {
            mfuLevels.get(oldLevel).hit(key);
        }

        experienceMap.put(key, newExp);
    }

    private void createNewEntry(String key, double exp) {
        if (experienceMap.size() >= capacity) {
            performEviction();
        }

        int level = determineLevel(exp);
        experienceMap.put(key, exp);
        mfuLevels.get(level).forceInsert(key, 1);
        aggregateLevelSum += level;
    }
    
    // --- Eviction Management ---

    private void performEviction() {
        recentEvictions++;
        
        List<String> candidates = gatherEvictionCandidates();
        String targetKey = findLeastRecentlyUsed(candidates);
        
        if (targetKey != null) {
            removeEntry(targetKey);
        }
    }

    private List<String> gatherEvictionCandidates() {
        List<String> candidates = new ArrayList<>();
        int levelIdx = 0;
        
        while (true) {
            SimpleMFU level = mfuLevels.get(levelIdx);
            if (level.getCurrentSize() > 0) {
                // For MFU, select from the highest frequency list
                String candidate = level.lists.get(level.max).iterator().next();
                candidates.add(candidate);
                
                // Random walk termination
                if (random.nextInt(1000) >= config.randomSkipThreshold) break;
            }
            
            levelIdx = (levelIdx + 1) % numLevels;
        }
        
        return candidates;
    }

    private String findLeastRecentlyUsed(List<String> candidates) {
        String oldestKey = null;
        int earliestTime = Integer.MAX_VALUE;

        for (String candidate : candidates) {
            int accessTime = lastAccessMap.getOrDefault(candidate, Integer.MAX_VALUE);
            if (accessTime < earliestTime) {
                earliestTime = accessTime;
                oldestKey = candidate;
            }
        }
        return oldestKey;
    }

    private void removeEntry(String key) {
        double exp = experienceMap.remove(key);
        lastAccessMap.remove(key);
        
        int level = determineLevel(exp);
        mfuLevels.get(level).remove(key);
        aggregateLevelSum -= level;
    }

    // --- Adaptive Mechanisms ---

    private void triggerAdaptiveCycle() {
        if (globalClock % adaptiveAgent.tuningFrequency == 0) {
            adaptiveAgent.performTuning(this);
        }
    }

    private void handleAging() {
        if (recentEvictions > config.agingThreshold) {
            config.ageMultiplier++;
            recentEvictions = 0;
        }
    }
    
    // ==================== Utility Methods ====================
    /**
     * Prints the distribution of entries across different cache levels.
     */
    public void printLevelDistribution() {
        int[] distribution = new int[numLevels];
        for (double exp : experienceMap.values()) {
            distribution[determineLevel(exp)]++;
        }
        
        System.out.println("--- Level Distribution ---");
        for (int i = 0; i < numLevels; i++) {
            if (distribution[i] > 0) {
                System.out.printf("Level %2d: %d entries%n", i, distribution[i]);
            }
        }
    }
    
    @Override
    public boolean request(String key) {
        // Single key request - not implemented for this multi-key cache
        return true;
    }
    
    @Override
    public int getCacheSize() {
        return capacity;
    }
    
    // ==================== Inner Classes ====================
    private static class GroupRequestResult {
        final List<Boolean> hitMissStatus;
        final List<Integer> levels;
        final int hits;

        GroupRequestResult(List<Boolean> status, List<Integer> levels, int hits) {
            this.hitMissStatus = status;
            this.levels = levels;
            this.hits = hits;
        }
    }

    private static class CacheConfig {
        int experienceFunction = 0; // 0: linear, 1: exponential
        double mmrRate = 0.3;
        int agingThreshold;
        int randomSkipThreshold = 50; // out of 1000
        int ageMultiplier = 100;

        CacheConfig(int capacity) {
            this.agingThreshold = (int) (capacity * 0.1);
        }
    }
    
    private static class QLearningAgent {
        // --- State/Action Space Config ---
        private static final int BUCKETS_FUNC = 2;
        private static final int BUCKETS_MMR = 10;
        private static final int BUCKETS_AGING = 10;
        private static final int BUCKETS_SKIP = 10;
        private static final int BUCKETS_LEVEL = 30;

        private static final int STATE_SPACE = BUCKETS_FUNC * BUCKETS_MMR * BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL;
        private static final int ACTION_SPACE = 54;

        // --- Agent State ---
        private final double[][] qTable = new double[STATE_SPACE][ACTION_SPACE];
        private final Random random = new Random();

        private int lastStateIdx = -1;
        private int lastActionIdx = -1;
        private int lastCheckpointHits = 0;
        private int lastPeriodGroupHits = 0;

        // --- Hyperparameters ---
        private double learningRate = 0.1;
        private double discountFactor = 0.99;
        private double explorationRate = 0.1;
        private final double explorationMin = 0.01;
        private final double explorationDecay = 0.001;

        final int tuningFrequency = 10000;

        void performTuning(VineCache_MFU cache) {
            int newStateIdx = computeStateIndex(cache);
            double reward = computeReward(cache);

            if (lastStateIdx != -1) {
                updateQValue(newStateIdx, reward);
            }

            int nextActionIdx = selectAction(newStateIdx);
            applyAction(nextActionIdx, cache.config);

            lastStateIdx = newStateIdx;
            lastActionIdx = nextActionIdx;
            explorationRate = Math.max(explorationMin, explorationRate - explorationDecay);
        }

        private int computeStateIndex(VineCache_MFU cache) {
            int fIdx = cache.config.experienceFunction;
            int mIdx = Math.min(9, (int) (cache.config.mmrRate / 0.1));
            int aIdx = Math.min(9, (int) (cache.config.agingThreshold / (cache.capacity * 0.01)));
            int sIdx = Math.min(9, cache.config.randomSkipThreshold / 10);
            int lIdx = cache.experienceMap.isEmpty() ? 0 : 
                       Math.min(29, cache.aggregateLevelSum / cache.experienceMap.size());

            return fIdx * (BUCKETS_MMR * BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL) +
                   mIdx * (BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL) +
                   aIdx * (BUCKETS_SKIP * BUCKETS_LEVEL) +
                   sIdx * BUCKETS_LEVEL +
                   lIdx;
        }

        private double computeReward(VineCache_MFU cache) {
            int currentHits = cache.groupHitCount - lastCheckpointHits;
            double reward = currentHits - lastPeriodGroupHits;

            lastCheckpointHits = cache.groupHitCount;
            lastPeriodGroupHits = currentHits;

            return reward;
        }
        
        private void updateQValue(int newStateIdx, double reward) {
            int nextBestAction = findBestAction(newStateIdx);
            double currentQ = qTable[lastStateIdx][lastActionIdx];
            double maxFutureQ = qTable[newStateIdx][nextBestAction];

            qTable[lastStateIdx][lastActionIdx] = currentQ + 
                learningRate * (reward + discountFactor * maxFutureQ - currentQ);
        }

        private int selectAction(int stateIdx) {
            if (random.nextDouble() < explorationRate) {
                return random.nextInt(ACTION_SPACE);
            }
            return findBestAction(stateIdx);
        }

        private int findBestAction(int stateIdx) {
            int bestIdx = 0;
            double maxVal = qTable[stateIdx][0];

            for (int i = 1; i < ACTION_SPACE; i++) {
                if (qTable[stateIdx][i] > maxVal) {
                    maxVal = qTable[stateIdx][i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
        
        private void applyAction(int action, CacheConfig config) {
            int funcAction = action / 27;
            int mmrAction = (action / 9) % 3;
            int agingAction = (action / 3) % 3;
            int skipAction = action % 3;
            
            // Apply function change
            if (funcAction == 1) {
                config.experienceFunction ^= 1;
            }
            
            // Apply MMR rate change
            double mmrStep = config.mmrRate * 0.05;
            if (mmrAction == 1) {
                config.mmrRate = Math.max(0.1, config.mmrRate - mmrStep);
            } else if (mmrAction == 2) {
                config.mmrRate = Math.min(1.0, config.mmrRate + mmrStep);
            }
            
            // Apply aging threshold change
            int agingStep = (int)(config.agingThreshold * 0.05);
            if (agingAction == 1) {
                config.agingThreshold = Math.max(0, config.agingThreshold - agingStep);
            } else if (agingAction == 2) {
                config.agingThreshold = Math.min((int)(config.agingThreshold * 2), config.agingThreshold + agingStep);
            }
            
            // Apply skip threshold change
            int skipStep = config.randomSkipThreshold / 20;
            if (skipAction == 1) {
                config.randomSkipThreshold = Math.max(0, config.randomSkipThreshold - skipStep);
            } else if (skipAction == 2) {
                config.randomSkipThreshold = Math.min(100, config.randomSkipThreshold + skipStep);
            }
        }
    }
}
