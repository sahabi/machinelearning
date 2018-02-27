{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module Models where

import Foreign.Storable (Storable)
import Numeric.LinearAlgebra ((<.>), (<>), (#>), inv, sumElements, Container)
import Numeric.LinearAlgebra.Data (vector, matrix, Matrix, (><), Vector, tr'
                                  ,cmap, toList, fromList, ident)

class Trainable m a b where
  train :: m a b -> [(a,b)] -> m a b

class Predictive m where
  predict :: m a b -> a -> b

vmap f = fromList . fmap f . toList

type LBInput a = [a]
type LabeledSet a = [(LBInput a, Double)]
type LBWeights = Vector Double
type LBFuncs a = [a -> Double]

data LBReg a b = LBReg { lbFuncs   :: LBFuncs (LBInput Double)
                       , lbWeights :: LBWeights
                       , lbError   :: Double
                       , lbReg     :: Double
                       , lbPredict :: LBWeights
                                   -> LBFuncs (LBInput Double)
                                   -> a
                                   -> b
                       }

instance Trainable LBReg [Double] Double where
  train = trainLBReg

instance Predictive LBReg where
  predict m = lbPredict m (lbWeights m) (lbFuncs m)

data OLBReg a b = OLBReg { olbFuncs   :: LBFuncs (LBInput Double)
                         , olbWeights :: LBWeights
                         , lrate      :: Double
                         , olbPredict :: LBWeights
                                      -> LBFuncs (LBInput Double)
                                      -> a
                                      -> b
                         }

instance Trainable OLBReg [Double] Double where
  train = foldl seqTrain

instance Predictive OLBReg where
  predict m = olbPredict m (olbWeights m) (olbFuncs m)

updateWeights x w e = x { lbWeights = w, lbError = e }
updateOWeights x w = x { olbWeights = w }

seqTrain :: OLBReg [Double] Double -> ([Double], Double) -> OLBReg [Double] Double
seqTrain m dp = updateOWeights m $
  w' + vmap (lrate m * (t - (w' <.> fi)) * ) fi where
    t = snd dp
    w' = olbWeights m
    fs = olbFuncs m
    fi = vector (zipWith ($) fs $ replicate (length fs) (fst dp))

trainLBReg :: LBReg [Double] Double -> [([Double], Double)] -> LBReg [Double] Double
trainLBReg m d = updateWeights m w e where
  t = vector (fmap snd d)
  fs = lbFuncs m
  dps = fmap fst d
  p = lbPredict m
  reg = lbReg m
  iden = ident 5 :: Matrix Double
  w = (inv (cmap (reg*) iden + (tr' dm <> dm)) <>  tr' dm) #> t where
    dm = tr' $ (length fs >< length dps) $ fs <*> dps
  e = sosError t pr where pr = vector $ fmap (p w fs) dps

sosError :: (Num (Vector a), Floating a, Container Vector a)
         => Vector a -> Vector a -> a
sosError t pr = sumElements $ vmap ((/2) . (**2)) $ t - pr

lbEval :: LBWeights -> LBFuncs (LBInput Double) -> LBInput Double -> Double
lbEval w fs d = w <.> vector (zipWith ($) fs $ replicate (length fs) d)

mkLBId :: Int -> LBFuncs (LBInput Double)
mkLBId n = const 1 : [(!!x) | x <- [0 .. n-1]]

mkLBReg :: LBFuncs (LBInput Double) -> LabeledSet Double -> LBReg (LBInput Double) Double
mkLBReg bfs = train LBReg { lbFuncs = bfs
                          , lbWeights = vector $ replicate (length bfs + 1) 1
                          , lbPredict = lbEval
                          , lbError = 999
                          , lbReg = 1
                          }

mkSLBReg :: Double -> LabeledSet Double -> LBReg (LBInput Double) Double
mkSLBReg reg ds = train LBReg { lbFuncs = mkLBId n
                           , lbWeights = vector $ replicate (n+1) 1
                           , lbPredict = lbEval
                           , lbError = 999
                           , lbReg = reg
                           } ds where n = length $ fst $ head ds

mkSOLBReg :: Double -> LabeledSet Double -> OLBReg (LBInput Double) Double
mkSOLBReg lr ds = train OLBReg { olbFuncs = mkLBId n
                            , olbWeights = vector $ replicate (n+1) 1
                            , olbPredict = lbEval
                            , lrate = lr
                            } ds where n = length $ fst $ head ds
