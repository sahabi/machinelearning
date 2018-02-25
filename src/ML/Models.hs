module Models where

import qualified Data.Vector as V
import qualified Data.Vector.Generic as G

-- ==================================================

class Trainable m where
  train :: m a b -> a -> m a b

class Predictive m where
  predict :: m a b -> a -> b


-- ===================================================
type LBInput = V.Vector Float

type LBWeights = V.Vector Float

type LBFuncs a = V.Vector (a -> Float)

data LBReg a b = LBReg { lbFuncs :: LBFuncs LBInput
                       , lbWeights :: LBWeights
                       , lbPredict :: LBWeights
                                   -> LBFuncs LBInput
                                   -> a
                                   -> b
                       }

instance Trainable LBReg where
  train m d = m

instance Predictive LBReg where
  predict m = lbPredict m (lbWeights m) (lbFuncs m)

mkLBReg :: LBFuncs LBInput -> LBReg Float Float
mkLBReg bfs = LBReg { lbFuncs = bfs
                    , lbWeights = V.fromList $ replicate (V.length bfs) 0
                    , lbPredict = \x y w -> 1 }

-- ==================================================
