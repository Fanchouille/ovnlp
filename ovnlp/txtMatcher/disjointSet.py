class DisjointSet(object):
    """
    Creates sets from pairs of ids : maps pairs to a disjoint set of networks and creates master id
    """

    def __init__(self, size=None):
        if size is None:
            self.leader = {}  # maps a member to the group's master
            self.group = {}  # maps a group master to the group (which is a set)
            self.oldgroup = {}
            self.oldleader = {}
        else:
            self.group = {i: set([i]) for i in range(0, size)}
            self.leader = {i: i for i in range(0, size)}
            self.oldgroup = {i: set([i]) for i in range(0, size)}
            self.oldleader = {i: i for i in range(0, size)}

    def add(self, a, b):
        self.oldgroup = self.group.copy()
        self.oldleader = self.leader.copy()
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb:
                    return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

    def connected(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                return leadera == leaderb
            else:
                return False
        else:
            return False

    def undo(self):
        self.group = self.oldgroup.copy()
        self.leader = self.oldleader.copy()
