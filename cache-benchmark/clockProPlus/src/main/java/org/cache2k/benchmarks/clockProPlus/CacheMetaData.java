package org.cache2k.benchmarks.clockProPlus;

@SuppressWarnings("WeakerAccess")
public class CacheMetaData {

  public boolean isReferenced() {
    return this.referenced;
  }

  public void setReference(boolean referenced) {
    this.referenced = referenced;
  }

  public void decAggHit() {
      this.aggHit--;
  }

  public void setAggHit(int aggHit) {
      this.aggHit = Math.max(this.aggHit, aggHit);
  }

  public int getAggHit() {
      return this.aggHit;
  }

  public void setHit(int hit) {
      this.hit = hit;
  }

  public int getHit() {
      return this.hit;
  }

  public void incHit() {
      this.hit++;
  }

  public void decHit() {
      this.hit--;
  }

  public boolean isEvCAR() {
      return this.evCAR;
  }
  
  public void setEvCARState(boolean evCAR) {
      this.evCAR = evCAR;
  }

  public boolean isLIR() {
    return this.lir;
  }

  public void setLIRState(boolean lir) {
    this.lir = lir;
  }

  public boolean isResident() {
    return this.resident;
  }

  public void demote(boolean demotion) {
    this.demoted = demotion;
  }

  public boolean isDemoted() {
    return this.demoted;
  }

  public void setResidentState(boolean resident) {
    this.resident = resident;
  }

  public void setAddress(String address) {
    this.address = address;
  }

  public String getAddress() {
    return this.address;
  }

  public void setNextAccessTime(int nextAccessTime) {
    this.nextAccessTime = nextAccessTime;
  }

  public int getNextAccessTime() {
    return this.nextAccessTime;
  }

  @Override
  public boolean equals(Object other){
    if (other == null) {
      return false;
    }
    if (other == this) {
      return true;
    }
    if (!(other instanceof CacheMetaData)) {
      return false;
    }
    CacheMetaData data = (CacheMetaData)other;
    if (data.getAddress() == this.getAddress()) {
      return true;
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return ((String)this.getAddress()).hashCode();
  }

  public CacheMetaData getNext() {
    return this.next;
  }

  public CacheMetaData getPrevious() {
    return this.previous;
  }

  public void linkNext(CacheMetaData next) {
    this.next = next;
    this.next.previous = this;
  }

  public void linkPrevious(CacheMetaData previous) {
    this.previous = previous;
    this.previous.next = this;
  }

  public void insertAfter(CacheMetaData target) {
    CacheMetaData tmpNext = target.getNext();
    target.linkNext(this);
    tmpNext.linkPrevious(this);
  }

  public void insertBefore(CacheMetaData target) {
    CacheMetaData tmpPrevious = target.getPrevious();
    target.linkPrevious(this);
    tmpPrevious.linkNext(this);
  }

  public void unlink() {
    CacheMetaData previous = this.previous;
    CacheMetaData next = this.next;
    previous.next = next;
    next.previous = previous;
    this.previous = null;
    this.next = null;
  }

  public void outputLinkedList() {
    CacheMetaData current = this;
    while (true) {
      String hot = "C";
      if (current.isLIR()) {
        hot = "H";
      }
      String resident = "";
      if (current.isResident()) {
        resident = "R";
      }
      String access = "";
      if (current.isReferenced()) {
        access = "A";
      }
      String outOfStack = "";
      if (!current.isInStack()) {
        outOfStack = "O";
      }
      System.out.printf("%d(%s%s%s%s), ", current.getAddress(), hot, resident, access, outOfStack);
      if (current.getPrevious().equals(this)) {
        break;
      } else {
        current = current.getPrevious();
      }
    }
    System.out.println();
  }

  public boolean isInStack() {
    return this.inStack;
  }

  public void setInStackStatus(boolean inStack) {
    this.inStack = inStack;
  }

  protected int aggHit;
  protected boolean evCAR;
  protected boolean lir;
  protected boolean resident;
  protected boolean demoted;
  protected String address;
  protected int nextAccessTime;
  protected boolean referenced;
  protected boolean inStack;
  protected CacheMetaData previous;
  protected CacheMetaData next;
  protected int hit;
}
