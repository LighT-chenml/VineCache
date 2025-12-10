package org.cache2k.benchmarks.clockProPlus;

import java.util.*;
import org.apache.commons.collections4.map.LinkedMap;

/**
 * VineCache_LRU: A multi-level cache with Q-learning based parameter tuning
 * Features:
 * - Hierarchical cache levels based on experience values
 * - Automatic parameter tuning using reinforcement learning
 * - Group-based caching with multi-key requests
 */
public class VineCache_LRU implements IMultiKeyCache {
    
    // ==================== Core Configuration ====================
    private static final int DEFAULT_LEVELS = 30;
    private static final double[] PRIORITY_RATES = {0.4, 0.8, 0.95};
    
    private final int capacity;
    private final int groupSize;
    private final int numLevels;
    private final int[] priorityThresholds;
    private final int[] levelThresholds;
    
    // ==================== Cache Data Structures ====================
    private final List<SimpleLRU> cacheLevels;
    private final Map<String, Double> experiences;
    private final Map<String, Integer> accessTimestamps;
    private final Random random;
    
    // ==================== Cache Statistics ====================
    private int currentTime;
    private int evictionCount;
    private int totalGroupHits;
    private int totalLevelSum;
    
    // ==================== Adaptive Parameters ====================
    private CacheConfig config;
    private QLearningAgent qAgent;
    
    // ==================== Constructor ====================
    public VineCache_LRU(int capacity, int groupSize) {
        this.capacity = capacity;
        this.groupSize = groupSize;
        this.numLevels = DEFAULT_LEVELS;
        this.random = new Random(19);
        
        // Initialize priority thresholds
        this.priorityThresholds = new int[PRIORITY_RATES.length];
        for (int i = 0; i < PRIORITY_RATES.length; i++) {
            priorityThresholds[i] = (int) (PRIORITY_RATES[i] * groupSize);
        }
        
        // Initialize level thresholds (exponential growth)
        this.levelThresholds = new int[numLevels - 1];
        levelThresholds[0] = 8;
        for (int i = 1; i < numLevels - 1; i++) {
            levelThresholds[i] = levelThresholds[i - 1] * 2;
        }
        
        // Initialize cache levels
        this.cacheLevels = new ArrayList<>();
        for (int i = 0; i < numLevels; i++) {
            cacheLevels.add(new SimpleLRU(capacity));
        }
        
        // Initialize data structures
        this.experiences = new HashMap<>();
        this.accessTimestamps = new HashMap<>();
        
        // Initialize adaptive components
        this.config = new CacheConfig();
        this.qAgent = new QLearningAgent();
        
        // Initialize statistics
        this.currentTime = 0;
        this.evictionCount = 0;
        this.totalGroupHits = 0;
        this.totalLevelSum = 0;
    }
    
    // ==================== Main Request Processing ====================
    @Override
    public List<Boolean> request(List<String> groupKeys) {
        currentTime++;
        
        // Process group request
        GroupRequestResult result = processGroupRequest(groupKeys);
        
        // Update cache based on results
        updateCacheFromGroupRequest(groupKeys, result);
        
        // Perform adaptive tuning if needed
        performAdaptiveTuning();
        
        return result.hitMissRecord;
    }
    
    private GroupRequestResult processGroupRequest(List<String> groupKeys) {
        List<Boolean> hitMissRecord = new ArrayList<>();
        List<Integer> levels = new ArrayList<>();
        int hitCount = 0;
        
        // Check hits/misses and record levels
        for (String key : groupKeys) {
            if (experiences.containsKey(key)) {
                hitMissRecord.add(true);
                hitCount++;
                levels.add(calculateLevel(experiences.get(key)));
            } else {
                hitMissRecord.add(false);
                levels.add(0);
            }
        }
        
        // Update group hit statistics
        if (hitCount == groupSize) {
            totalGroupHits++;
        }
        
        return new GroupRequestResult(hitMissRecord, levels, hitCount);
    }
    
    private void updateCacheFromGroupRequest(List<String> groupKeys, GroupRequestResult result) {
        // Calculate experience deltas
        double baseExperience = calculateBaseExperience(result.hitCount);
        List<Double> experienceDeltas = calculateExperienceDeltas(result.levels, baseExperience);
        
        // Update each key
        for (int i = 0; i < groupKeys.size(); i++) {
            updateKey(groupKeys.get(i), experienceDeltas.get(i));
        }
        
        // Handle aging
        handleAging();
    }
    
    // ==================== Experience Calculation ====================
    private double calculateBaseExperience(int hitCount) {
        double experience = calculateAggregateHitExperience(hitCount);
        return experience * config.ageMultiplier / 100.0;
    }
    
    private double calculateAggregateHitExperience(int hitCount) {
        if (config.experienceFunction == 0) {
            return calculateLinearExperience(hitCount);
        } else {
            return Math.pow(Math.E, (hitCount - 10) / 2.0);
        }
    }
    
    private double calculateLinearExperience(int hitCount) {
        if (hitCount < priorityThresholds[0]) return 0;
        if (hitCount < priorityThresholds[1]) return hitCount - priorityThresholds[0];
        if (hitCount < priorityThresholds[2]) {
            return priorityThresholds[1] - priorityThresholds[0] + 
                   (hitCount - priorityThresholds[1]) * 2;
        }
        return priorityThresholds[1] - priorityThresholds[0] + 
               (priorityThresholds[2] - priorityThresholds[1]) * 2 + 
               (hitCount - priorityThresholds[2]) * 8;
    }
    
    private List<Double> calculateExperienceDeltas(List<Integer> levels, double baseExperience) {
        List<Double> deltas = new ArrayList<>();
        
        for (int i = 0; i < levels.size(); i++) {
            int currentLevel = levels.get(i);
            double mmr = calculateMMR(levels, currentLevel);
            deltas.add(baseExperience * mmr);
        }
        
        return deltas;
    }
    
    private double calculateMMR(List<Integer> levels, int currentLevel) {
        int sum = 0;
        for (int level : levels) {
            int diff = level - currentLevel;
            sum += diff * diff * diff;  // Cubic difference
        }
        
        double mmr = 0;
        if (sum > 0) mmr = Math.log(sum);
        else if (sum < 0) mmr = -Math.log(-sum);
        
        mmr *= config.mmrRate;
        return sigmoid(mmr);
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }
    
    // ==================== Cache Level Management ====================
    private int calculateLevel(double experience) {
        for (int i = 0; i < numLevels - 1; i++) {
            if (experience <= levelThresholds[i]) {
                return i;
            }
        }
        return numLevels - 1;
    }
    
    private void updateKey(String key, double experienceDelta) {
        if (experiences.containsKey(key)) {
            updateExistingKey(key, experienceDelta);
        } else {
            insertNewKey(key, experienceDelta);
        }
        accessTimestamps.put(key, currentTime);
    }
    
    private void updateExistingKey(String key, double experienceDelta) {
        double oldExperience = experiences.get(key);
        double newExperience = oldExperience + experienceDelta;
        int oldLevel = calculateLevel(oldExperience);
        int newLevel = calculateLevel(newExperience);
        
        if (oldLevel != newLevel) {
            // Move between levels
            cacheLevels.get(oldLevel).remove(key);
            cacheLevels.get(newLevel).forceInsert(key);
            totalLevelSum += newLevel - oldLevel;
        } else {
            // Update within same level
            cacheLevels.get(oldLevel).hit(key);
        }
        
        experiences.put(key, newExperience);
    }
    
    private void insertNewKey(String key, double experience) {
        if (experiences.size() >= capacity) {
            evictKey();
        }
        
        int level = calculateLevel(experience);
        experiences.put(key, experience);
        cacheLevels.get(level).forceInsert(key);
        totalLevelSum += level;
    }
    
    // ==================== Eviction Logic ====================
    private void evictKey() {
        evictionCount++;
        
        // Find candidates from different levels with random skip
        List<String> candidates = findEvictionCandidates();
        
        // Select the least recently used among candidates
        String evictKey = selectLRUCandidate(candidates);
        
        // Remove from cache
        removeKey(evictKey);
    }
    
    private List<String> findEvictionCandidates() {
        List<String> candidates = new ArrayList<>();
        
        for (int i = 0; ; i = (i == numLevels - 1) ? 0 : i + 1) {
            if (cacheLevels.get(i).getCurrentSize() == 0) continue;
            
            String candidate = cacheLevels.get(i).lru.firstKey();
            candidates.add(candidate);
            
            // Random skip mechanism
            if (random.nextInt(1000) >= config.randomSkipThreshold) {
                break;
            }
        }
        
        return candidates;
    }
    
    private String selectLRUCandidate(List<String> candidates) {
        String selected = null;
        int earliestTime = Integer.MAX_VALUE;
        
        for (String candidate : candidates) {
            int accessTime = accessTimestamps.get(candidate);
            if (accessTime < earliestTime) {
                earliestTime = accessTime;
                selected = candidate;
            }
        }
        
        return selected;
    }
    
    private void removeKey(String key) {
        double experience = experiences.get(key);
        int level = calculateLevel(experience);
        
        cacheLevels.get(level).evict();
        experiences.remove(key);
        accessTimestamps.remove(key);
        totalLevelSum -= level;
    }
    
    // ==================== Adaptive Tuning ====================
    private void performAdaptiveTuning() {
        handleAging();
        
        if (currentTime % qAgent.tuningFrequency == 0) {
            qAgent.performTuning(this);
        }
    }
    
    private void handleAging() {
        if (evictionCount > config.agingThreshold) {
            config.ageMultiplier++;
            evictionCount = 0;
        }
    }
    
    // ==================== Utility Methods ====================
    public void printLevelDistribution() {
        int[] distribution = new int[numLevels];
        for (double experience : experiences.values()) {
            distribution[calculateLevel(experience)]++;
        }
        
        for (int i = 0; i < numLevels; i++) {
            System.out.println("Level " + i + ": " + distribution[i]);
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
        final List<Boolean> hitMissRecord;
        final List<Integer> levels;
        final int hitCount;
        
        GroupRequestResult(List<Boolean> hitMissRecord, List<Integer> levels, int hitCount) {
            this.hitMissRecord = hitMissRecord;
            this.levels = levels;
            this.hitCount = hitCount;
        }
    }
    
    private static class CacheConfig {
        int experienceFunction = 0;          // 0: linear, 1: exponential
        double mmrRate = 0.3;                // MMR calculation rate
        int agingThreshold;                  // Threshold for aging
        int randomSkipThreshold = 50;        // Random skip probability (out of 1000)
        int ageMultiplier = 100;             // Age multiplier for experience
        
        CacheConfig() {
            // agingThreshold will be set based on capacity
        }
        
        void initialize(int capacity) {
            this.agingThreshold = (int) (capacity * 0.1);
        }
    }
    
    private static class QLearningAgent {
        // Q-Learning parameters
        private static final int FUNC_BUCKETS = 2;
        private static final int MMR_BUCKETS = 10;
        private static final int AGING_BUCKETS = 10;
        private static final int SKIP_BUCKETS = 10;
        private static final int LEVEL_BUCKETS = 30;
        private static final int STATE_SPACE = FUNC_BUCKETS * MMR_BUCKETS * AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS;
        private static final int ACTION_SPACE = 54;
        
        private final double[][] qTable;
        private final Random random = new Random();
        
        private int currentState = -1;
        private int currentAction = -1;
        private int lastGroupHits = 0;
        private int lastPeriodHits = 0;
        
        // Learning parameters
        private double learningRate = 0.1;
        private double discountFactor = 0.99;
        private double explorationRate = 0.1;
        private final double minExplorationRate = 0.01;
        private final double explorationDecay = 0.001;
        
        final int tuningFrequency = 10000;
        
        QLearningAgent() {
            this.qTable = new double[STATE_SPACE][ACTION_SPACE];
            // Q-table is initialized to zeros by default
        }
        
        void performTuning(VineCache_LRU cache) {
            int newState = calculateState(cache);
            double reward = calculateReward(cache);
            
            if (currentState != -1) {
                updateQValue(newState, reward);
            }
            
            int action = selectAction(newState);
            applyAction(action, cache.config);
            
            currentState = newState;
            currentAction = action;
            
            // Decay exploration rate
            explorationRate = Math.max(minExplorationRate, explorationRate - explorationDecay);
        }
        
        private int calculateState(VineCache_LRU cache) {
            int funcBucket = cache.config.experienceFunction;
            int mmrBucket = Math.min(9, (int)(cache.config.mmrRate / 0.1));
            int agingBucket = Math.min(9, (int)(cache.config.agingThreshold / (cache.capacity * 0.01)));
            int skipBucket = Math.min(9, cache.config.randomSkipThreshold / 10);
            int levelBucket = cache.experiences.isEmpty() ? 0 : 
                             Math.min(29, cache.totalLevelSum / cache.experiences.size());
            
            return funcBucket * (MMR_BUCKETS * AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS) + 
                   mmrBucket * (AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS) + 
                   agingBucket * (SKIP_BUCKETS * LEVEL_BUCKETS) +
                   skipBucket * LEVEL_BUCKETS +
                   levelBucket;
        }
        
        private double calculateReward(VineCache_LRU cache) {
            int currentGroupHits = cache.totalGroupHits - lastGroupHits;
            double reward = currentGroupHits - lastPeriodHits;
            
            lastGroupHits = cache.totalGroupHits;
            lastPeriodHits = currentGroupHits;
            
            return reward;
        }
        
        private void updateQValue(int newState, double reward) {
            int bestNextAction = getBestAction(newState);
            double currentQ = qTable[currentState][currentAction];
            double maxNextQ = qTable[newState][bestNextAction];
            
            qTable[currentState][currentAction] = currentQ + 
                learningRate * (reward + discountFactor * maxNextQ - currentQ);
        }
        
        private int selectAction(int state) {
            if (random.nextDouble() < explorationRate) {
                return random.nextInt(ACTION_SPACE);
            } else {
                return getBestAction(state);
            }
        }
        
        private int getBestAction(int state) {
            int bestAction = 0;
            double bestValue = qTable[state][0];
            
            for (int i = 1; i < ACTION_SPACE; i++) {
                if (qTable[state][i] > bestValue) {
                    bestValue = qTable[state][i];
                    bestAction = i;
                }
            }
            
            return bestAction;
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
