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
