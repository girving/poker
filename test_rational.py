#!/usr/bin/env python

from __future__ import division
from numpy import *
from rational import *

R = rational

def test_misc():
    x = R()
    y = R(7) 
    z = R(-6,-10)
    assert not x
    assert y and z
    assert z.n is 3
    assert z.d is 5
    assert str(y)=='7'
    assert str(z)=='3/5'
    assert repr(y)=='rational(7)'
    assert repr(z)=='rational(3,5)'

def test_compare():
    random.seed(1262081)
    for _ in xrange(100):
        xn,yn = random.randint(-10,10,2)
        xd,yd = random.randint(1,10,2)
        x,y = R(xn,xd),R(yn,yd)
        assert bool(x)==bool(xn)
        assert (x==y)==(xn*yd==yn*xd)
        assert (x<y)==(xn*yd<yn*xd)
        assert (x>y)==(xn*yd>yn*xd)
        assert (x<=y)==(xn*yd<=yn*xd)
        assert (x>=y)==(xn*yd>=yn*xd)
        # Not true in general, but should be for this sample size
        assert (hash(x)==hash(y))==(x==y)

def test_arithmetic():
    random.seed(1262081)
    for _ in xrange(100):
        xn,yn,zn = random.randint(-100,100,3)
        xd,yd,zd = [n if n else 1 for n in random.randint(-100,100,3)]
        x,y,z = R(xn,xd),R(yn,yd),R(zn,zd)
        assert -x==R(-xn,xd)
        assert +x is x
        assert --x==x
        assert x+y==R(xn*yd+yn*xd,xd*yd)
        assert x+y==x--y==R(xn*yd+yn*xd,xd*yd)
        assert -x+y==-(x-y)
        assert (x+y)+z==x+(y+z)
        assert x*y==R(xn*yn,xd*yd)
        assert (x*y)*z==x*(y*z)
        assert -(x*y)==(-x)*y
        assert x*y==y*x
        assert x*(y+z)==x*y+x*z
        if y:
            assert x/y==R(xn*yd,xd*yn)
            assert x/y*y==x
            assert x//y==xn*yd//(xd*yn)
            assert x%y==x-x//y
        assert x+7==7+x==x+R(7)
        assert x*7==7*x==x*R(7)
        assert int(x)==int(xn/xd)
        assert allclose(float(x),xn/xd)
        assert abs(x)==R(abs(xn),abs(xd))
        # TODO: test floor, ceil, abs

def test_errors():
    # Check invalid constructions
    for args in (R(3,2),4),(1.2,),(1,2,3):
        try:
            R(*args)
            assert False
        except TypeError:
            pass
    for args in (1<<80,),(2,1<<80):
        try:
            R(*args)
            assert False
        except OverflowError:
            pass
    # Check for zero divisions
    try:
        R(1,0)
        assert False
    except ZeroDivisionError:
        pass
    try:
        R(7)/R()
        assert False
    except ZeroDivisionError:
        pass
    # Check for LONG_MIN overflows
    for args in (-1<<63,-1),(1<<63,):
        try:
            R(*args)
            assert False
        except OverflowError:
            pass
    # Check for overflow in addition
    r = R(1<<62)
    try:
        r+r
        assert False
    except OverflowError:
        pass
    # Check for overflow in multiplication
    p = R(1262081,1262083) # Twin primes from http://primes.utm.edu/lists/small/10ktwins.txt
    r = p
    for _ in xrange(int(log(2.**63)/log(r.d))-1):
        r *= p
    try:
        r*p
        assert False
    except OverflowError:
        pass
    # Float/rational arithmetic should fail
    for x,y in (.2,R(3,2)),(R(3,2),.2):
        try:
            x+y
            assert False
        except TypeError:
            pass

def test_numpy_basic():
    d = dtype(rational)
    assert d.itemsize==16
    x = zeros(5,d)
    assert type(x[2]) is rational
    assert x[3]==0
    assert ones(5,d)[3]==1
    x[2] = 2
    assert x[2]==2
    x[3] = R(4,5)
    assert 5*x[3]==4
    try:
        x[4] = 1.2
        assert False
    except TypeError:
        pass
    i = arange(R(1,3),R(5,3),R(1,3))
    assert i.dtype is d
    print i.dtype,array([R(1,3),R(2,3),R(3,3),R(4,3)]).dtype
    print equal(i,i)
    print i==[R(1,3),R(2,3),R(3,3),R(4,3)]
    print i==array([R(1,3),R(2,3),R(3,3),R(4,3)])
    assert all(i==[R(1,3),R(2,3),R(3,3),R(4,3)])
    y = zeros(4,d)
    y[1:3] = i[1:3] # Test unstride copyswapn
    assert all(y==[0,R(2,3),R(3,3),0])
    assert all(nonzero(y)[0]==(1,2))
    y[::3] = i[:2] # Test strided copyswapn
    assert all(y==[R(1,3),R(2,3),R(3,3),R(2,3)])
    assert searchsorted(arange(0,20),R(7,2))==4 # Test compare
    assert argmax(y)==2 # Test argmax
    assert dot(i,y)==R(22,9)
    y[:] = 7 # Test fillwithscalar
    assert all(y==7)

def test_numpy_cast():
    r = arange(R(10,3),step=R(1,3),dtype=rational)
    array([3],dtype=int64)*array([R(5,3)],dtype=rational)
    for T in int32,int64:
        n = arange(10,dtype=T)
        assert all(n.astype(rational)==3*r)
        print n+r
        print 4*r
        assert all(n+r==4*r)
    for T in float,double:
        f = arange(10,dtype=float)/3
        assert allclose(r.astype(float),f)
        assert allclose(r+f,2*f)
        try:
            f.astype(rational)
            assert False
        except ValueError:
            pass

def test_numpy_arithmetic():
    d = dtype(rational)
    assert (3*arange(10,dtype=d)).dtype is d
