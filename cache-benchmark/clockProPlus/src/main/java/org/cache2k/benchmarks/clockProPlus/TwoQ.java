package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

public class TwoQ implements ISimpleCache {

    protected int windowSize;
    protected int mainCacheSize;
    protected SimpleFIFO window;
    protected SimpleLRU mainCache;

    public TwoQ(int capacity) {
        this.windowSize = (int)(capacity * 0.1);
        this.mainCacheSize = capacity - this.windowSize;
        this.window = new SimpleFIFO(this.windowSize);
        this.mainCache = new SimpleLRU(this.mainCacheSize);
    }

    @Override
    public boolean request(String key) {
        if (this.window.exists(key)) {
            this.window.remove(key);
            this.mainCache.forceInsert(key);
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
