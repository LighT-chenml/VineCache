package org.cache2k.benchmarks.clockProPlus;

import java.util.*;

public class SimpleMFU implements ISimpleCache{
    TreeSet<Integer> countSet;
    HashMap<String, String> vals;// cache K and V
    HashMap<String, Integer> counts;// K and counters
    HashMap<Integer, LinkedHashSet<String>> lists;// Counter and item list
    int cap;
    public int size;
    int request_number = 0;
    int max;

    public SimpleMFU(int capacity) {
        cap = capacity;
        this.size = capacity;
        vals = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        countSet = new TreeSet<>();
        max = -1;
    }

    @Override
    public boolean request(String key) {
        if (get(key)) {
            return true;
        } else {
            set(key, key);
            return false;
        }
    }

    public void hit(String key) {
        // Get the count from counts map
        int count = counts.get(key);

        // System.out.println("hit " + key + " count = " + count + " max = " + max);

        // increase the counter
        counts.put(key, count + 1);
        // remove the element from the counter to linkedhashset
        lists.get(count).remove(key);
        if (lists.get(count).size() == 0) {
            lists.remove(count);
            countSet.remove(count);
        }
        if (!lists.containsKey(count + 1))
        {
            lists.put(count + 1, new LinkedHashSet<>());
            countSet.add(count + 1);
        }
        lists.get(count + 1).add(key);
        max = Math.max(max, count + 1);
    }

    public boolean exists(String key) {
        return vals.containsKey(key);
    }

    public boolean get(String key) {
        if (!vals.containsKey(key))
            return false;
        hit(key);
        // return vals.get(key);
        return true;
    }

    public void set(String key, String value) {
        if (cap <= 0)
            return;
        // If key does exist, we are returning from here
        if (vals.containsKey(key)) {
            vals.put(key, value);
            hit(key);
            return;
        }
        forceInsert(key, 1);
    }

    public String evict() {
        String evit = lists.get(max).iterator().next();
        remove(evit);
        return evit;
    }

    int remove(String key) {
        int count = counts.remove(key);
        lists.get(count).remove(key);
        vals.remove(key);
        if (lists.get(count).size() == 0) {
            lists.remove(count);
            countSet.remove(count);
            if (countSet.size() == 0) max = -1;
            else if (count == max) max = countSet.last();
        }
        return count;
    }

    public void forceInsert(String key, int count) {
        if (vals.size() >= cap) {
            evict();
        }
        vals.put(key, key);
        counts.put(key, count);
        if (!lists.containsKey(count))
        {
            lists.put(count, new LinkedHashSet<>());
            countSet.add(count);
        }
        lists.get(count).add(key);
        max = Math.max(max, count);
    }

    public int getCurrentSize() {
        return vals.size();
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }
}
