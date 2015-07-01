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

typedef double v4df __attribute__((vector_size(32)));
static inline v4df v4_broadcast(double d) {
  v4df foo = {d, d, d, d};
  return foo;
}

typedef double v2df __attribute__((vector_size(16)));
static inline v2df v2_broadcast(double d) {
  v2df foo = {d, d};
  return foo;
}
