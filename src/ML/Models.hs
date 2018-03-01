{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module Models where

import Foreign.Storable (Storable)
import Numeric.LinearAlgebra ((<.>), (<>), (#>), inv, sumElements, Container)
import Numeric.LinearAlgebra.Data (vector, matrix, Matrix, (><), Vector, tr'
                                  ,cmap, ident, size, fromRows)

class Trainable m a b where
  train :: m a b -> [(a,b)] -> m a b

class Predictive m where
  predict :: m a b -> a -> b

type LBWeights = Matrix Double
type LBFuncs a = [a -> Double]

data LBReg a b = LBReg { lbFuncs   :: LBFuncs a
                       , lbWeights :: LBWeights
                       , lbError   :: Double
                       , lbReg     :: Double
                       , lbPredict :: LBWeights
                                   -> LBFuncs a
                                   -> a
                                   -> b
                       }

instance Trainable LBReg [a] (Vector Double) where
  train = trainLBReg

instance Predictive LBReg where
  predict m = lbPredict m (lbWeights m) (lbFuncs m)

updateWeights x w e = x { lbWeights = w, lbError = e }

trainLBReg :: LBReg [a] (Vector Double) -> [([a], Vector Double)] -> LBReg [a] (Vector Double)
trainLBReg m d = updateWeights m w e where
  t    = fromRows (fmap snd d)
  fs   = lbFuncs m
  dps  = fmap fst d
  p    = lbPredict m
  reg  = lbReg m
  iden = ident $ 1 + (length . head) dps :: Matrix Double
  dm   = tr' $ (length fs >< length dps) $ fs <*> dps
  w    = (inv (cmap (reg*) iden + (tr' dm <> dm)) <>  tr' dm) <> t
  e    = 2--sosError t pr where pr = vector $ fmap (p w fs) dps

sosError :: Vector Double -> Vector Double -> Double
sosError t pr = 0--sumElements $ cmap ((/2) . (**2)) $ t - pr

lbEval :: LBWeights -> LBFuncs a -> a -> Vector Double
lbEval w fs d = tr' w #> vector (zipWith ($) fs $ replicate (length fs) d)

mkLBId :: Int -> LBFuncs [Double]
mkLBId n = const 1 : [(!!x) | x <- [0 .. n-1]]

mkSLBReg :: Double -> [([Double], Vector Double)] -> LBReg [Double] (Vector Double)
mkSLBReg reg ds = train LBReg { lbFuncs   = mkLBId n
                              , lbWeights = (n><k) [1 .. ]
                              , lbPredict = lbEval
                              , lbError   = fromIntegral (maxBound :: Int)
                              , lbReg     = reg
                              } ds where n = length $ fst $ head ds
                                         k = size $ snd $ head ds
