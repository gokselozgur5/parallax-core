//! Circular buffer Key-Value cache implementation with boundary wrap-around safety.

/// Error conditions encountered during circular buffer KV cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheError {
    /// Provided dimension was zero or overflowed memory calculations.
    InvalidDimension,
    /// Provided buffer slice length did not match expected dimensions.
    BufferLengthMismatch,
    /// Requested sequence position was out of available token bounds.
    IndexOutOfBounds,
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "invalid dimension or arithmetic overflow"),
            Self::BufferLengthMismatch => write!(f, "buffer length mismatch for KV cache shape"),
            Self::IndexOutOfBounds => write!(f, "token index out of bounds in circular cache"),
        }
    }
}

impl std::error::Error for KvCacheError {}

/// In-place circular buffer KV cache manager with boundary wrap-around safety.
#[derive(Debug, Clone)]
pub struct CircularKvBuffer {
    /// Number of attention heads for key/value.
    pub num_heads: usize,
    /// Head dimension size.
    pub head_dim: usize,
    /// Maximum circular token capacity.
    pub capacity: usize,
    /// Total lifetime tokens ingested (monotonic counter).
    pub total_inserted: usize,
}

impl CircularKvBuffer {
    /// Create a new circular KV buffer descriptor.
    pub fn new(num_heads: usize, head_dim: usize, capacity: usize) -> Result<Self, KvCacheError> {
        if num_heads == 0 || head_dim == 0 || capacity == 0 {
            return Err(KvCacheError::InvalidDimension);
        }
        let per_token = num_heads
            .checked_mul(head_dim)
            .ok_or(KvCacheError::InvalidDimension)?;
        let _ = capacity
            .checked_mul(per_token)
            .ok_or(KvCacheError::InvalidDimension)?;
        Ok(Self {
            num_heads,
            head_dim,
            capacity,
            total_inserted: 0,
        })
    }

    /// Number of valid active tokens currently stored in the circular cache.
    #[inline]
    pub fn active_tokens(&self) -> usize {
        self.total_inserted.min(self.capacity)
    }

    /// Slot index in the circular buffer [0, capacity) where the next token is written.
    #[inline]
    pub fn write_cursor(&self) -> usize {
        self.total_inserted % self.capacity
    }

    /// Convert a relative logical token index in [0, active_tokens) to physical slot index.
    pub fn logical_to_physical(&self, logical_idx: usize) -> Result<usize, KvCacheError> {
        let active = self.active_tokens();
        if logical_idx >= active {
            return Err(KvCacheError::IndexOutOfBounds);
        }
        if self.total_inserted < self.capacity {
            Ok(logical_idx)
        } else {
            // Buffer has wrapped: oldest token is at write_cursor()
            let phys = (self.write_cursor() + logical_idx) % self.capacity;
            Ok(phys)
        }
    }

    /// Calculate buffer offset and length for a given (head, physical_slot).
    #[inline]
    pub fn slot_offset(&self, head: usize, slot: usize) -> Result<usize, KvCacheError> {
        if head >= self.num_heads || slot >= self.capacity {
            return Err(KvCacheError::InvalidDimension);
        }
        // Offset: (head * capacity + slot) * head_dim
        let h_offset = head
            .checked_mul(self.capacity)
            .ok_or(KvCacheError::InvalidDimension)?;
        let slot_offset = h_offset
            .checked_add(slot)
            .ok_or(KvCacheError::InvalidDimension)?;
        slot_offset
            .checked_mul(self.head_dim)
            .ok_or(KvCacheError::InvalidDimension)
    }

    /// Total expected buffer capacity in float elements (num_heads * capacity * head_dim).
    #[inline]
    pub fn required_buffer_len(&self) -> Result<usize, KvCacheError> {
        let per_head = self
            .capacity
            .checked_mul(self.head_dim)
            .ok_or(KvCacheError::InvalidDimension)?;
        self.num_heads
            .checked_mul(per_head)
            .ok_or(KvCacheError::InvalidDimension)
    }

    /// Insert single key-value token slices [num_heads, head_dim] into circular cache.
    pub fn append(
        &mut self,
        k_buf: &mut [f32],
        v_buf: &mut [f32],
        new_k: &[f32],
        new_v: &[f32],
    ) -> Result<usize, KvCacheError> {
        let req_buf = self.required_buffer_len()?;
        if k_buf.len() != req_buf || v_buf.len() != req_buf {
            return Err(KvCacheError::BufferLengthMismatch);
        }
        let per_token = self
            .num_heads
            .checked_mul(self.head_dim)
            .ok_or(KvCacheError::InvalidDimension)?;
        if new_k.len() != per_token || new_v.len() != per_token {
            return Err(KvCacheError::BufferLengthMismatch);
        }
        let slot = self.write_cursor();
        for h in 0..self.num_heads {
            let offset = self.slot_offset(h, slot)?;
            let src_start = h * self.head_dim;
            let src_end = src_start + self.head_dim;
            k_buf[offset..offset + self.head_dim].copy_from_slice(&new_k[src_start..src_end]);
            v_buf[offset..offset + self.head_dim].copy_from_slice(&new_v[src_start..src_end]);
        }
        self.total_inserted = self
            .total_inserted
            .checked_add(1)
            .ok_or(KvCacheError::InvalidDimension)?;
        Ok(slot)
    }

    /// Read key and value vectors for a given head and logical index in time order.
    pub fn read_token(
        &self,
        k_buf: &[f32],
        v_buf: &[f32],
        head: usize,
        logical_idx: usize,
        out_k: &mut [f32],
        out_v: &mut [f32],
    ) -> Result<(), KvCacheError> {
        let req_buf = self.required_buffer_len()?;
        if k_buf.len() != req_buf || v_buf.len() != req_buf {
            return Err(KvCacheError::BufferLengthMismatch);
        }
        if out_k.len() != self.head_dim || out_v.len() != self.head_dim {
            return Err(KvCacheError::BufferLengthMismatch);
        }
        let phys_slot = self.logical_to_physical(logical_idx)?;
        let offset = self.slot_offset(head, phys_slot)?;
        out_k.copy_from_slice(&k_buf[offset..offset + self.head_dim]);
        out_v.copy_from_slice(&v_buf[offset..offset + self.head_dim]);
        Ok(())
    }

    /// Reset circular buffer counters back to empty state.
    #[inline]
    pub fn clear(&mut self) {
        self.total_inserted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_invalid_dimensions() {
        assert_eq!(CircularKvBuffer::new(0, 64, 128).unwrap_err(), KvCacheError::InvalidDimension);
        assert_eq!(CircularKvBuffer::new(4, 0, 128).unwrap_err(), KvCacheError::InvalidDimension);
        assert_eq!(CircularKvBuffer::new(4, 64, 0).unwrap_err(), KvCacheError::InvalidDimension);
        assert_eq!(CircularKvBuffer::new(usize::MAX, 64, 128).unwrap_err(), KvCacheError::InvalidDimension);
    }

    #[test]
    fn test_kv_cache_append_and_read_no_wrap() {
        let mut cache = CircularKvBuffer::new(2, 4, 3).unwrap();
        let mut k_buf = vec![0.0f32; cache.required_buffer_len().unwrap()];
        let mut v_buf = vec![0.0f32; cache.required_buffer_len().unwrap()];

        let k1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let slot = cache.append(&mut k_buf, &mut v_buf, &k1, &v1).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(cache.active_tokens(), 1);

        let mut out_k = [0.0f32; 4];
        let mut out_v = [0.0f32; 4];
        cache.read_token(&k_buf, &v_buf, 0, 0, &mut out_k, &mut out_v).unwrap();
        assert_eq!(out_k, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out_v, [10.0, 20.0, 30.0, 40.0]);
        cache.read_token(&k_buf, &v_buf, 1, 0, &mut out_k, &mut out_v).unwrap();
        assert_eq!(out_k, [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(out_v, [50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_kv_cache_circular_wrap_around_boundary_safety() {
        let mut cache = CircularKvBuffer::new(1, 2, 3).unwrap();
        let mut k_buf = vec![0.0f32; 6];
        let mut v_buf = vec![0.0f32; 6];
        for i in 0..3 {
            let t = i as f32;
            let slot = cache.append(&mut k_buf, &mut v_buf, &[t, t + 0.5], &[t * 10.0, t * 10.0 + 5.0]).unwrap();
            assert_eq!(slot, i);
        }
        assert_eq!(cache.active_tokens(), 3);
        assert_eq!(cache.write_cursor(), 0);

        let slot4 = cache.append(&mut k_buf, &mut v_buf, &[3.0, 3.5], &[30.0, 35.0]).unwrap();
        assert_eq!(slot4, 0);
        assert_eq!(cache.active_tokens(), 3);
        assert_eq!(cache.write_cursor(), 1);
        assert_eq!(cache.total_inserted, 4);

        let mut out_k = [0.0f32; 2];
        let mut out_v = [0.0f32; 2];
        cache.read_token(&k_buf, &v_buf, 0, 0, &mut out_k, &mut out_v).unwrap();
        assert_eq!(out_k, [1.0, 1.5]);
        assert_eq!(out_v, [10.0, 15.0]);
        cache.read_token(&k_buf, &v_buf, 0, 2, &mut out_k, &mut out_v).unwrap();
        assert_eq!(out_k, [3.0, 3.5]);
        assert_eq!(out_v, [30.0, 35.0]);

        assert_eq!(cache.read_token(&k_buf, &v_buf, 0, 3, &mut out_k, &mut out_v).unwrap_err(), KvCacheError::IndexOutOfBounds);
    }

    #[test]
    fn test_kv_cache_buffer_mismatch_and_clear() {
        let mut cache = CircularKvBuffer::new(1, 2, 2).unwrap();
        let mut k_buf = vec![0.0f32; 4];
        let mut v_buf = vec![0.0f32; 3]; // invalid size

        assert_eq!(cache.append(&mut k_buf, &mut v_buf, &[1.0, 2.0], &[3.0, 4.0]).unwrap_err(), KvCacheError::BufferLengthMismatch);

        let mut v_buf_valid = vec![0.0f32; 4];
        assert_eq!(cache.append(&mut k_buf, &mut v_buf_valid, &[1.0], &[3.0, 4.0]).unwrap_err(), KvCacheError::BufferLengthMismatch);

        assert_eq!(cache.append(&mut k_buf, &mut v_buf_valid, &[1.0, 2.0], &[3.0, 4.0]).unwrap(), 0);
        assert_eq!(cache.active_tokens(), 1);

        cache.clear();
        assert_eq!(cache.active_tokens(), 0);
        assert_eq!(cache.total_inserted, 0);
    }
}
