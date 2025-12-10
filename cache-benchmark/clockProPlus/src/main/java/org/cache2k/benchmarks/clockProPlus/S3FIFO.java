package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

public class S3FIFO implements ISimpleCache {

    protected int FIFOSize;
    protected int FIFOGhostSize;
    protected int mainCacheSize;
    protected SimpleFIFO FIFO;
    protected SimpleFIFO FIFOGhost;
    protected ClockNBit mainCache;

    public S3FIFO(int capacity) {
        this.FIFOSize = (int) (capacity * 0.1);
        this.mainCacheSize = capacity - this.FIFOSize;
        this.FIFOGhostSize = this.mainCacheSize;
        this.FIFO = new SimpleFIFO(this.FIFOSize);
        this.FIFOGhost = new SimpleFIFO(this.FIFOGhostSize);
        this.mainCache = new ClockNBit(this.mainCacheSize, 3);
    }

    @Override
    public boolean request(String key) {
        if (this.FIFO.exists(key)) {
            // hit FIFO
            this.FIFO.hit(key);
            return true;
        } else if (this.mainCache.exists(key)) {
            // hit main cache
            this.mainCache.hit(key);
            return true;
        } else {
            // miss
            if (this.FIFOGhost.exists(key)) {
                this.FIFOGhost.remove(key);
                this.mainCache.forceInsert(key);
            } else {
                if (this.FIFO.isFull()) {
                    CacheMetaData data = this.FIFO.evict();
                    if (data.isReferenced()) {
                        this.mainCache.forceInsert(data.getAddress());
                    } else {
                        this.FIFOGhost.forceInsert(data.getAddress());
                    }
                }
                this.FIFO.insert(key);
            }
            return false;
        }
    }

    @Override
    public int getCacheSize() {
        return this.FIFOSize + this.mainCacheSize;
    }
}
