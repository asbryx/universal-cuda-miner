#include "../cuda/range.hpp"
#include <cassert>
#include <iostream>
int main() {
  for(unsigned w=1;w<=32;++w) {
    uint64_t s[4]={},o[4];
    for(unsigned b=0;b<w;++b)s[b/8]|=uint64_t(255)<<(8*(b%8));
    nonce_range::validate(s,1,1,w);
    bool rejected=false;
    try {nonce_range::validate(s,2,1,w);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
    s[0]--;
    nonce_range::validate(s,2,1,w);
    nonce_range::advance(s,1,1,o);
    assert(o[0]==s[0]+1);
  }
  uint64_t s[4]={},o[4];
  nonce_range::advance(s,UINT64_MAX,UINT64_MAX,o);
  assert(o[0]==1 && o[1]==UINT64_MAX-1 && o[2]==0 && o[3]==0);
  std::cout<<"all 32 widths and 128-bit multiplication passed\n";
}
