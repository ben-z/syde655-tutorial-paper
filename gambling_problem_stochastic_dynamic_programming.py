# Below implementation is derived from
# https://en.wikipedia.org/wiki/Stochastic_dynamic_programming#cite_note-1
# and
# https://github.com/gwr3n/jsdp/blob/master/jsdp/src/main/java/jsdp/app/standalone/stochastic/GamblersRuin.java
from typing import List, Tuple
import memoize as mem
import functools 

class memoize: 
    
    def __init__(self, func): 
        self.func = func 
        self.memoized = {} 
        self.method_cache = {} 

    def __call__(self, *args): 
        return self.cache_get(self.memoized, args, 
            lambda: self.func(*args)) 

    def __get__(self, obj, objtype): 
        return self.cache_get(self.method_cache, obj, 
            lambda: self.__class__(functools.partial(self.func, obj))) 

    def cache_get(self, cache, key, func): 
        try: 
            return cache[key] 
        except KeyError: 
            cache[key] = func() 
            return cache[key] 
    
    def reset(self):
        self.memoized = {} 
        self.method_cache = {} 

class State:
    '''the state of the gambler's ruin problem
    '''

    def __init__(self, t: int, wealth: float):
        '''state constructor
        
        Arguments:
            t {int} -- time period
            wealth {float} -- initial wealth
        '''
        self.t, self.wealth = t, wealth

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.t) + " " + str(self.wealth)

    def __hash__(self):
        return hash(str(self))

class GamblersRuin:

    def __init__(self, bettingHorizon:int, targetWealth: float, pmf: List[List[Tuple[int, float]]]):
        '''the gambler's ruin problem
        
        Arguments:
            bettingHorizon {int} -- betting horizon
            targetWealth {float} -- target wealth
            pmf {List[List[Tuple[int, float]]]} -- probability mass function
        '''

        # initialize instance variables
        self.bettingHorizon, self.targetWealth, self.pmf = bettingHorizon, targetWealth, pmf

        # lambdas
        self.ag = lambda s: [i for i in range(0, min(self.targetWealth//2, s.wealth) + 1)] # action generator
        self.st = lambda s, a, r: State(s.t + 1, s.wealth - a + a*r)                       # state transition
        self.iv = lambda s, a, r: 1 if s.wealth - a + a*r >= self.targetWealth else 0      # immediate value function

        self.cache_actions = {}  # cache with optimal state/action pairs

    def f(self, wealth: float) -> float:
        s = State(0, wealth)
        return self._f(s)

    def q(self, t: int, wealth: float) -> float:
        s = State(t, wealth)
        return self.cache_actions[str(s)]

    @memoize
    def _f(self, s: State) -> float:
        #Forward recursion
        v = max(
            [sum([p[1]*(self._f(self.st(s, a, p[0])) 
                  if s.t < self.bettingHorizon - 1 else self.iv(s, a, p[0]))   # future value
                  for p in self.pmf[s.t]])                                     # random variable realisations
             for a in self.ag(s)])                                             # actions

        opt_a = lambda a: sum([p[1]*(self._f(self.st(s, a, p[0])) 
                               if s.t < self.bettingHorizon - 1 else self.iv(s, a, p[0])) 
                               for p in self.pmf[s.t]]) == v          
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        
        return v                                                                # return value

if __name__ == '__main__':
    instance = {"bettingHorizon": 4, "targetWealth": 6, "pmf": [[(0, 0.6),(2, 0.4)] for i in range(0,4)]}
    gr, initial_wealth = GamblersRuin(**instance), 2

    # f_1(x) is gambler's probability of attaining $targetWealth at the end of bettingHorizon
    print("f_1("+str(initial_wealth)+"): " + str(gr.f(initial_wealth))) 

    #Recover optimal action for period 2 when initial wealth at the beginning of period 2 is $1.
    t, initial_wealth = 1, 1
    print("b_"+str(t+1)+"("+str(initial_wealth)+"): " + str(gr.q(t, initial_wealth)))
