"""All byte-pair codes, finite scale codes, non-contiguous inputs and refusal."""
import unittest
import numpy as np
from repack import VARIANTS


def oracle(b):
    # Decode only integer nibble indices in ggml logical order, then adjacent pack.
    codes=np.concatenate((b[:,1:]&15,b[:,1:]>>4),axis=1)
    return codes[:,0::2]|(codes[:,1::2]<<4),b[:,0].copy()


class RepackTests(unittest.TestCase):
    def check(self,b):
        snapshot=b.copy();expected=oracle(b)
        for name,fn in VARIANTS.items():
            with self.subTest(variant=name,shape=b.shape,strides=b.strides):
                w,s=fn(b)
                np.testing.assert_array_equal(w,expected[0])
                np.testing.assert_array_equal(s,expected[1])
                np.testing.assert_array_equal(b,snapshot)
                self.assertFalse(np.shares_memory(w,b))
                self.assertFalse(np.shares_memory(s,b))
                self.assertTrue(w.flags.c_contiguous)
                self.assertTrue(s.flags.c_contiguous)

    def test_all_byte_pairs_all_finite_scales(self):
        x=np.arange(65536,dtype=np.uint32)
        b=np.empty((len(x),17),dtype=np.uint8)
        b[:,0]=x%255
        b[:,1::2]=(x&255)[:,None]
        b[:,2::2]=(x>>8)[:,None]
        self.check(b)

    def test_random_strided_reversed_and_readonly(self):
        rng=np.random.default_rng(1017)
        b=rng.integers(0,256,(1031,17),dtype=np.uint8);b[:,0]%=255
        self.check(b)
        self.check(b[::-1])
        self.check(b[::3])
        wide=np.zeros((len(b),34),dtype=np.uint8);wide[:,::2]=b
        self.check(wide[:,::2])
        b.flags.writeable=False
        self.check(b)

    def test_empty_and_single(self):
        self.check(np.empty((0,17),dtype=np.uint8))
        self.check(np.full((1,17),128,dtype=np.uint8))

    def test_refuses_nan_and_invalid_geometry(self):
        b=np.zeros((3,17),dtype=np.uint8)
        for fn in VARIANTS.values():
            for index in (0,2):
                bad=b.copy();bad[index,0]=255
                with self.assertRaises(ValueError):fn(bad)
            for bad in (b.astype(np.int16),b[:,:16],np.zeros(17,dtype=np.uint8)):
                with self.assertRaises(ValueError):fn(bad)


if __name__=='__main__':unittest.main()
