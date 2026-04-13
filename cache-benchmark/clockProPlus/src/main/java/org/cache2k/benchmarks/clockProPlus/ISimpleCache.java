package org.cache2k.benchmarks.clockProPlus;

import org.cache2k.benchmark.EvictionStatistics;

public interface ISimpleCache {
    int getCacheSize();
    
    boolean request(String address);

    default EvictionStatistics getEvictionStatistics() {
        return new EvictionStatistics() {
        };
    }

}
