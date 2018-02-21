module LinearBasis where

import qualified Data.Vector.Unboxed as U
import qualified Data.Vector as V
import qualified Data.Vector.Generic as G

type Input = V.Vector Float
type Output = V.Vector Float

newtype BasisFunction = BasisFunction { func :: Input -> Float }

f' :: Input -> Output
f' = undefined

linearBasis :: V.Vector BasisFunction -> Input -> Float
linearBasis f i = G.sum x where x =  G.map func f <*> G.replicate (G.length f) i


