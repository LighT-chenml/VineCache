package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

public class LRUK implements ISimpleCache {

    protected int windowSize;
    protected int mainCacheSize;
    protected int frequencyThreshold;
    protected SimpleLRU window;
    protected SimpleLRU mainCache;

    public LRUK(int capacity, int K) {
        this.frequencyThreshold = K-1;
        this.windowSize = (int)(capacity * 0.3);
        this.mainCacheSize = capacity - this.windowSize;
        this.window = new SimpleLRU(this.windowSize);
        this.mainCache = new SimpleLRU(this.mainCacheSize);
    }

    @Override
    public boolean request(String key) {
        if (this.window.exists(key)) {
            CacheMetaData data = this.window.get(key);
            data.incHit();
            if (data.getHit() >= this.frequencyThreshold) {
                this.window.remove(key);
                this.mainCache.forceInsert(key);
            }
            else {
                this.window.hit(key);
            }
            return true;
        }
        else if (this.mainCache.exists(key)) {
            this.mainCache.hit(key);
            return true;
        }
        else {
            this.window.forceInsert(key);
            return false;
        }
    }

    @Override
    public int getCacheSize() {
        return this.windowSize + this.mainCacheSize;
    }
}
