package org.cache2k.benchmarks.clockProPlus;

import org.apache.commons.collections4.map.LinkedMap;

public class ClockNBit implements ISimpleCache {

    protected LinkedMap<String, CacheMetaData> clock;
    protected int size;
    protected int hitLimit;

    public ClockNBit(int size, int hitLimit) {
        this.hitLimit = hitLimit;
        this.size = size;
        this.clock = new LinkedMap<String, CacheMetaData>();
    }

    @Override
    public boolean request(String address) {
        return request(address, 0);
    }

    public void hit(String address) {
        CacheMetaData data = this.clock.get(address);
        data.incHit();
    }

    public boolean isFull() {
        return this.size <= this.clock.size();
    }

    public void forceInsert (String key) {
        if (!this.isFull()) {
            this.insert(key);
        } else {
            this.replace(key);
        }
    }

    public boolean request(String address, int aggHit) {
        if (this.exists(address)) {
            CacheMetaData data = this.clock.get(address);
            int clockHit = data.getHit();
            if (clockHit < hitLimit) data.incHit();
            if (data.getAggHit() < aggHit) {
                // For EvCAR logic
                data.setEvCARState(true);
                data.setAggHit(aggHit);
            }
            return true;
        }
        if (this.size > this.clock.size()) {
            this.insert(address, aggHit);
        } else {
            this.replace(address);
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuffer str = new StringBuffer(String.format("Clock(%d)", this.size));
        str.append(this.clock.asList().toString());
        return str.toString();
    }

    public int getCurrentSize() {
        return this.clock.size();
    }

    protected boolean exists(String address) {
        return this.clock.containsKey(address);
    }

    protected void move() {
        String firstKey = this.clock.firstKey();
        CacheMetaData data = this.clock.remove(firstKey);
        // move the first key to the end of the list
        this.clock.put(firstKey, data);
    }

    public void remove(String key) {
        this.clock.remove(key);
    }

    protected void insert(String address) {
        this.insert(address, 0);
    }
    
    protected void insert(String address, int aggHit) {
        if (this.size < this.clock.size()) {
            System.err.println("Error at clock insert; the clock size is over the limit");
        }
        CacheMetaData data = new CacheMetaData();
        data.setAddress(address);
        data.setAggHit(aggHit);
        data.setHit(0);
        // System.out.println("data referenced " + data.referenced);
        this.clock.put(address, data);
        // this.checkSize();
    }

    protected CacheMetaData check() {
        String firstKey = this.clock.firstKey();
        return this.clock.get(firstKey);
    }

    protected String evict() {
        String firstKey = this.clock.firstKey();
        this.clock.remove(firstKey);
        return firstKey;
    }

    protected void replace(String address) {
        while (true) {
            CacheMetaData data = this.check();
            if (data.getHit() > 0) {
                data.decHit();
            } else {
                this.evict();
                this.insert(address);
                break;
            }
            this.move();
        }
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }

    protected void checkSize() {
    }
}
