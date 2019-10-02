#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from utils import fig2hex

def __main__():
    print 'This module is not intended for direct command line usage. Currently suports import to Python only.'
    return

class htmlElement:
    def __init__(self, uid, args):
        self.uid = uid
        if 'eleName' in args:
            self.eleName = args['eleName']
        else:
            self.eleName = 'div'
        if 'class' in args:
            self.eclass = args['class']
        else:
            self.eclass = None
        if 'stored' in args:
            self.stored = True
            self.stemp = args['stored']
        else:
            self.stored = False
        if 'innerHTML' in args:
            self.innerHTML = args['innerHTML']
        else:
            self.innerHTML = ''
        if 'id' in args:
            self.imid = args['id']
        else:
            self.imid = uid
        if 'figure' in args:
            self.fig = args['figure']
        else:
            self.fig = None
        if 'figargs' in args:
            self.figargs = args['figargs']
        else:
            self.figargs = None
        if 'children' in args:
            self.children = htmlFrame(prepop=args['children'])
        else:
            self.children = htmlFrame()
        self.labels = {}
        for k, v in args.iteritems():
            if k not in ['eleName', 'stored', 'innerHTML', 'children', 'figure', 'figargs', 'sidelabel']:
                self.labels[k] = v
        return
    
    def append(self, e):
        self.children.append(e)
        return
    
    def html(self, inner_only=False, figargs=[True, None, 72]):
        labels = ''
        if self.figargs != None:
            for i in range(0, len(self.figargs)):
                if self.figargs[i] != None:
                    figargs[i] = self.figargs[i]
        for k, v in self.labels.iteritems():
            labels += ' ' + k + '="' + v + '"'
        h = '<'+self.eleName+labels+'>\n'
        if inner_only:
            h = ''
        if self.stored:
            f = open(self.stemp, 'r')
            t = f.read()
            f.close()
            os.remove(self.stemp)
        else:
            t = self.innerHTML
            if self.fig != None:
                if figargs[1] == None:
                    d = os.getcwd()+'/'
                else:
                    d = figargs[1]
                self.fig.savefig(d+self.imid+'.pdf')
                if figargs[0]:
                    t += fig2hex(self.fig, dpi=figargs[2]) + '\n'
                else:
                    t += '<embed src="img/'+self.imid+'.pdf" type="application/pdf">\n'
                #self.fig.close() !close them somehow...
            for child in self.children:
                if isinstance(child, dict):
                    child = htmlElement(**child)
                t += child.html(figargs=figargs)
        h += t
        if inner_only:
            return h
        else:
            return h + '\n</'+self.eleName+'>\n'

class htmlFrame:
    def __init__(self, prepop=None):
        self.children = {}
        self.order = []
        if prepop != None:
            for a in prepop:
                i = a.pop('uid', None)
                self.append(htmlElement(i, a))
        return
    
    def __len__(self):
        return len(self.order)
    
    def __contains__(self, key):
        if key in self.order: 
            return True
        else:
            for k, v in self.children.iteritems():
                if key in v.children:
                    return True
                else:
                    return False

    def __getitem__(self, key):
        if key in self.children:
            return self.children[key]
        else:
            for k, v in self.children.iteritems():
                if key in v.children:
                    return self.children[k].children[key]
    
    def __iter__(self):
        for e in self.order:
            yield self.children[e]
    
    def append(self, e):
        self.order.append(e.uid)
        self.children[e.uid] = e
        return

class htmlReporter:
    def __init__(self, fn, template=None, embedpics=True, savepics=True, saveres=200):
        self.fn = fn
        self.embedpics = embedpics
        self.saveres = saveres
        if savepics:
            self.imgDir = os.path.dirname(fn)+'/img/'
            if not os.path.exists(self.imgDir):
                os.makedirs(self.imgDir)
        if template == None:
            self.head, self.feet = '<html>\n<head></head>\n<body>\n', '</body>\n</html>'
        else:
            f = open(template, 'r')
            t = f.read()
            self.head, self.feet = t.split('***CONTENT***')
            f.close()
        self.members = htmlFrame()
        self.N = 0
        return
    
    def append(self, name, div={}, tempstore=False, parent=None, sidelabel=None):
        self.N += 1
        element = htmlElement(name, div)
        if tempstore:
            divname = self.fn+'_tmp_'+str(self.N)+'_div'
            f = open(divname, 'w')
            f.write(element.html(inner_only=True, figargs=[self.embedpics, self.imgDir, self.saveres]))
            f.close()
            element.stored = True
            element.stemp = divname
            element.innerHTML = ''
        if parent == None:
            parent = self.members
        else:
            if parent in self.members:
                parent = self.members[parent]
            else:
                parent = self.members
        parent.append(element)
        if sidelabel != None:
            sidebar, itemname, baritem = sidelabel
            if sidebar in self.members:
                self.members[sidebar].append(htmlElement(itemname, baritem))
        return
    
    def add(self, name, div={}, **kwargs):
        if 'div' not in kwargs:
            kwargs['div'] = div
        if isinstance(name, list):
            for en, ek in name:
                self.append(en, **ek)
        else:
            self.append(name, **kwargs)
        return
    
    def report(self):
        o = self.head + '\n'
        for e in self.members:
            o += e.html(figargs=[self.embedpics, self.imgDir, self.saveres]) + '\n'                
        o += self.feet
        f = open(self.fn, 'w')
        f.write(o)
        f.close()
        return

if __name__ == '__main__':
    __main__()