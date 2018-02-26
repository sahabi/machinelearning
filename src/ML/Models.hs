{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module Models where

import Foreign.Storable (Storable)
import Numeric.LinearAlgebra ((<.>), (<>), (#>), pinv, sumElements, Container)
import Numeric.LinearAlgebra.Data (vector, matrix, Matrix, (><), Vector, tr'
                                  , toList, fromList)

class Trainable m a b where
  train :: m a b -> [(a,b)] -> m a b

class Predictive m where
  predict :: m a b -> a -> b

vmap f = fromList . fmap f . toList

type LBInput a = [a]
type LabeledSet a = [(LBInput a, Double)]
type LBWeights = Vector Double
type LBFuncs a = [a -> Double]

data LBReg a b = LBReg { lbFuncs :: LBFuncs (LBInput Double)
                       , lbWeights :: LBWeights
                       , lbPredict :: LBWeights
                                   -> LBFuncs (LBInput Double)
                                   -> a
                                   -> b
                       , lbError :: Double
                       }

instance Trainable LBReg [Double] Double where
  train = trainLBReg

instance Predictive LBReg where
  predict m = lbPredict m (lbWeights m) (lbFuncs m)

updateWeights x w e = x { lbWeights = w, lbError = e }

trainLBReg :: LBReg [Double] Double -> [([Double], Double)] -> LBReg [Double] Double
trainLBReg m d = updateWeights m w e where
  t = vector (fmap snd d)
  fs = lbFuncs m
  dps = fmap fst d
  p = lbPredict m
  w = pinv dm #> t where
    dm = tr' $ (length fs >< length dps) $ fs <*> dps
  e = sosError t pr where pr = vector $ fmap (p w fs) dps

sosError :: (Num (Vector a), Floating a, Container Vector a) => Vector a -> Vector a -> a
sosError t pr = sumElements $ vmap ((/2) . (**2)) $ t - pr

lbEval :: LBWeights -> LBFuncs (LBInput Double) -> LBInput Double -> Double
lbEval w fs d = w <.> vector (zipWith ($) fs $ replicate (length fs) d)

mkLBId :: Int -> LBFuncs (LBInput Double)
mkLBId n = const 1 : [(!!x) | x <- [0 .. n-1]]

mkLBReg :: LBFuncs (LBInput Double) -> LabeledSet Double -> LBReg (LBInput Double) Double
mkLBReg bfs = train LBReg { lbFuncs = bfs
                          , lbWeights = vector $ replicate (length bfs + 1) 0
                          , lbPredict = lbEval
                          , lbError = 999
                          }

mkSLBReg :: LabeledSet Double -> LBReg (LBInput Double) Double
mkSLBReg  ds = train LBReg { lbFuncs = mkLBId n
                           , lbWeights = vector $ replicate (n+1) 0
                           , lbPredict = lbEval
                           , lbError = 999
                           } ds where n = length $ fst $ head ds
