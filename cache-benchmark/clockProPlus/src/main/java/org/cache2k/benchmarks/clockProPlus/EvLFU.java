package org.cache2k.benchmarks.clockProPlus;

import java.util.*;

public class EvLFU implements IMultiKeyCache {

    int nEVTable; // nEVTable
    HashMap<String, Integer> vals;// cache Key and aggHit ; we ignore the value here
    ArrayList<HashSet<String>> lists;// Counter and item list
    int min = 0;
    public int size;
    int maxPerfectItem = 0;
    int request_number = 0;

    /*
    * Add flushing:
    */
    int nPerfectItem = 0;
    double flushRate = 0.1;
    double perfectItemCapacity = 0.9;

    public EvLFU(int capacity, int nGroup) {
        this.size = capacity;
        this.nEVTable = nGroup;
        vals = new HashMap<>();
        lists = new ArrayList<>();
        for (int idx = 0; idx <= nEVTable; idx++) {
            lists.add( new LinkedHashSet<>());
        }
        maxPerfectItem = (int) (this.size * perfectItemCapacity);
    }

    public Boolean exists(String key) {
        return vals.containsKey(key);
    }

    public void outputSize() {
        for (int i=0;i<lists.size();++i)
            System.out.println(i + " " + lists.get(i).size());
    }

    @Override
    public List<Boolean> request(List<String> groupKeys) {
        List<Boolean> aggHitMissRecord = new ArrayList<Boolean>();
        int aggHit = 0;
        for (String key : groupKeys) {
            if (vals.containsKey(key)) {
                aggHitMissRecord.add(true);
                aggHit++;
            } else {
                aggHitMissRecord.add(false);
            }
        }
        for (String key : groupKeys) {
            update(key, aggHit);
        }

        if (aggHit == nEVTable) {
            nPerfectItem = lists.get(nEVTable).size();
        }

        // request_number++;
        // if (request_number % 10000 == 0)
        // {
        //     System.out.println(request_number);
        //     outputSize();
        // }

        return aggHitMissRecord;
    }

    // Updating the existing keys and inserting the missing keys
    void update(String key, int aggHit) {
        boolean isHit = updateAggHit(key, aggHit);
        if (!isHit) {
            // On Miss
            forceInsert(key, aggHit);
        }
    }

    // update the aggHit value of the current key
    public boolean updateAggHit(String key, int aggHit) {
        Integer oldAggHit = vals.get(key);
        if (oldAggHit != null) {
            if (oldAggHit < aggHit) {
                // update the old agg_hit
                lists.get(oldAggHit).remove(key);
                lists.get(aggHit).add(key);
                vals.put(key, aggHit);
            }
            // Hit
            return true;
        } else {
            // Miss
            return false;
        }
    }
    
    // Inserting the NEW key
    public void forceInsert(String key, int aggHit) {
        /*
        * Flushing:
        */
        nPerfectItem = lists.get(nEVTable).size();
        if (nPerfectItem >= maxPerfectItem) {
            // System.out.println("do flushing!");
            for (int i = 0; i < flushRate * lists.get(nEVTable).size(); i++) {
              String keyToEvict = lists.get(nEVTable).iterator().next();
              lists.get(nEVTable).remove(keyToEvict);
              vals.remove(keyToEvict);
            }
            // adjust the nPerfectItem counter
            nPerfectItem = lists.get(nEVTable).size();
        } else if (vals.size() >= this.size) {
            // cache is full
            while(lists.get(min).isEmpty()) {
                // find the right key to pop
                // Update minimum agg_hit
                min += 1;
                if (min > this.nEVTable){
                    min = 1;
                }
            }
            String keyToEvict = lists.get(min).iterator().next();
            lists.get(min).remove(keyToEvict);
            vals.remove(keyToEvict);
        }
        // insert the new value:
        vals.put(key, aggHit);
        lists.get(aggHit).add(key);

        if (aggHit < min) {
            min = aggHit;
        }
    }

    @Override
    public boolean request(String key) {
        if (updateAggHit(key, 0)) {
            return true;
        } else {
            forceInsert(key, 0);
            return false;
        }
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }

}
