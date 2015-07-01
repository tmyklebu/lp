/*
Tor Myklebust's LP solver
Copyright (C) 2013-2015 Tor Myklebust (tmyklebu@csclub.uwaterloo.ca)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "lp.h"

static vector<int *> vmi, vni;
static vector<double *> vmd, vnd;
static int last_m, last_n;

static void clean_m() {
  while (vmi.size()) {
    delete[] vmi.back(); 
    vmi.pop_back();
  }
  while (vmd.size()) {
    delete[] vmd.back(); 
    vmd.pop_back();
  }
}

static void clean_n() {
  while (vni.size()) {
    delete[] vni.back(); 
    vni.pop_back();
  }
  while (vnd.size()) {
    delete[] vnd.back(); 
    vnd.pop_back();
  }
}

vecmi::vecmi() {
  if (m != last_m) clean_m();
  if (vmi.size()) {
    p = vmi.back();
    vmi.pop_back();
  } else p = new int[m];
}

vecmi::~vecmi() {
  vmi.push_back(p);
}

vecni::vecni() {
  if (n != last_n) clean_n();
  if (vni.size()) {
    p = vni.back();
    vni.pop_back();
  } else p = new int[n];
}

vecni::~vecni() {
  vni.push_back(p);
}

vecmd::vecmd() {
  if (m != last_m) clean_m();
  if (vmd.size()) {
    p = vmd.back();
    vmd.pop_back();
  } else p = new double[m];
}

vecmd::~vecmd() {
  vmd.push_back(p);
}

vecnd::vecnd() {
  if (n != last_n) clean_n();
  if (vnd.size()) {
    p = vnd.back();
    vnd.pop_back();
  } else p = new double[n];
}

vecnd::~vecnd() {
  vnd.push_back(p);
}

void allocation_cleanup() {
  m = n = -1;
  clean_m(); clean_n();
}
