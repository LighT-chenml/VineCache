package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

@SuppressWarnings("WeakerAccess")
public class SimpleFIFO implements ISimpleCache {

    public SimpleFIFO(int size) {
        this.size = size;
        this.fifoList = new LinkedMap<String, CacheMetaData>();
    }

    public boolean isFull() {
        return this.fifoList.size() >= this.size;
    }

    public void insert(String address) {
        CacheMetaData data = new CacheMetaData();
        data.setAddress(address);
        data.setReference(false);
        this.fifoList.put(address, data);
    }

    public void insert(String address, CacheMetaData data) {
        this.fifoList.put(address, data);
    }

    public void hit(String address) {
        CacheMetaData data = this.fifoList.get(address);
        data.setReference(true);
    }

    public CacheMetaData get(String address) {
        return this.fifoList.get(address);
    }

    public void forceInsert (String key) {
        if (this.isFull()) {
            this.evict();
        }
        this.insert(key);
    }

    @Override
    public boolean request(String address) {
        if (!this.fifoList.containsKey(address)) {
            if (this.fifoList.size() >= this.size) {
                this.evict();
            }
            insert(address);
            return false;
        } else {
            return true;
        }
    }

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer(String.format("FIFO(%d)", this.size));
        str.append(this.fifoList.asList().toString());
        return str.toString();
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }

    public int getCurrentSize() {
        return this.fifoList.size();
    }

    public boolean exists(String key) {
        return this.fifoList.containsKey(key);
    }

    public CacheMetaData evict() {
        String firstKey = this.fifoList.firstKey();
        return this.fifoList.remove(firstKey);
    }

    public void remove(String key) {
        this.fifoList.remove(key);
    }

    protected int size;
    protected LinkedMap<String, CacheMetaData> fifoList;

}
