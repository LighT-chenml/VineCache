package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

@SuppressWarnings("WeakerAccess")
public class SimpleLRU implements ISimpleCache {

    public SimpleLRU(int size) {
        this.size = size;
        this.lru = new LinkedMap<String, CacheMetaData>();
    }

    @Override
    public synchronized boolean request(String address) {
        if (!this.lru.containsKey(address)) {
            forceInsert(address);
            return false;
        } else {
            hit(address);
            return true;
        }
    }

    public boolean exists(String address) {
        return this.lru.containsKey(address);
    }

    public CacheMetaData get(String address) {
        return this.lru.get(address);
    }

    public int getCurrentSize() {
        return this.lru.size();
    }

    public void remove(String address) {
        this.lru.remove(address);
    }

    public void hit(String address) {
        CacheMetaData data = this.lru.get(address);
        this.lru.remove(address);
        this.lru.put(address, data);
    }

    public String evict() {
        String firstKey = this.lru.firstKey();
        this.lru.remove(firstKey);
        return firstKey;
    }

    public void forceInsert(String address) {
        if (this.lru.size() >= this.size) {
            evict();
        }
        this.lru.put(address, new CacheMetaData());
    }

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer(String.format("LRU(%d)", this.size));
        str.append(this.lru.asList().toString());
        return str.toString();
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }

    protected int size;
    protected LinkedMap<String, CacheMetaData> lru;

}
