#pragma once

namespace minerva
{
  
  enum ArithmeticType {
    ADD = 0,
    SUB,
    MULT,
    DIV,
  };

  enum ElewiseType {
    EXP = 0,
    LN,
    SIGMOID,
    NEGATIVE,
  };

  enum ReductionType {
    SUM = 0,
    MAX,
  };
 
}
