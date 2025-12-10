package org.cache2k.benchmarks.clockProPlus;

import java.util.*;

public class SimpleLFU implements ISimpleCache{
    HashMap<String, String> vals;// cache K and V
    HashMap<String, Integer> counts;// K and counters
    HashMap<Integer, LinkedHashSet<String>> lists;// Counter and item list
    int cap;
    public int size;
    int min = -1;
    int request_number = 0;

    public SimpleLFU(int capacity) {
        cap = capacity;
        this.size = capacity;
        vals = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        lists.put(1, new LinkedHashSet<>());
    }

    public void output() {
        int sum=0;
        int sum0=0;
        int sum1=0;
        int sum2=0;
        int sum3=0;
        int sum4=0;
        for (int i=0;sum<lists.size();++i)
        {
            if (lists.get(i) == null) continue;
            int s = lists.get(i).size();
            if (s == 0) continue;
            sum += lists.get(i).size();
            if (i<10) sum0+=s;
            else if (i<100) sum0+=s;
            else if (i<1000) sum1+=s;
            else if (i<10000) sum1+=s;
            else sum2+=s;
        }
        System.out.println(0 + " " + sum0);
        System.out.println(1 + " " + sum1);
        System.out.println(2 + " " + sum2);
        // System.out.println(3 + " " + sum3);
        // System.out.println(4 + " " + sum4);
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
        // increase the counter
        counts.put(key, count + 1);
        // remove the element from the counter to linkedhashset
        lists.get(count).remove(key);

        // when current min does not have any data, next one would be the min
        if (count == min && lists.get(count).size() == 0)
            min++;
        if (!lists.containsKey(count + 1))
            lists.put(count + 1, new LinkedHashSet<>());
        lists.get(count + 1).add(key);
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
        forceInsert(key);
    }

    public void evict() {
        while (lists.get(min).size() == 0)
            min++;
        String evit = lists.get(min).iterator().next();
        lists.get(min).remove(evit);
        vals.remove(evit);
        counts.remove(evit);
    }

    public void forceInsert(String key) {
        if (vals.size() >= cap) {
            String evit = lists.get(min).iterator().next();
            lists.get(min).remove(evit);
            vals.remove(evit);
            counts.remove(evit);
        }
        vals.put(key, key);
        counts.put(key, 1);
        min = 1;
        lists.get(1).add(key);
    }

    public int getCurrentSize() {
        return vals.size();
    }

    @Override
    public int getCacheSize() {
        return this.size;
    }
}
