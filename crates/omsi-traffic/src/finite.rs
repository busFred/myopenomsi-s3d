//! JSON has no infinity: serde_json writes an infinite float as `null`, which then does
//! not read back as a number. The traffic has some (a way through a junction that meets
//! nobody has its first conflict at infinity), so a recording writes its records through
//! [`Finite`], which turns them into numbers that read back the same with [`from_str`]:
//! ±1e39 for an `f32` (beyond its range, it reads back as infinity) and ±`f64::MAX` for an
//! `f64` (which has no such number). A NaN stays `null`: it is a bug, and the line will
//! not read.

use serde::de::DeserializeOwned;
use serde::ser::{self, Serialize, Serializer};

/// Read what [`to_string`] wrote. (The exact reading refuses a number beyond an `f32`'s
/// range for one, so a line with an infinity is read through a `Value`, which takes the
/// number as an `f64` and rounds it to the `f32` the field is.)
pub fn from_str<T: DeserializeOwned>(s: &str) -> serde_json::Result<T> {
    serde_json::from_str(s).or_else(|e| serde_json::from_str::<serde_json::Value>(s).and_then(serde_json::from_value).map_err(|_| e))
}

/// The text of `v` as JSON, infinities kept (see the module).
pub fn to_string<T: Serialize + ?Sized>(v: &T) -> serde_json::Result<String> {
    let mut out = Vec::with_capacity(256);
    v.serialize(Finite(&mut serde_json::Serializer::new(&mut out)))?;
    // (serde_json writes UTF-8 only)
    Ok(String::from_utf8(out).unwrap_or_default())
}

/// Write `v` as JSON into `w`, infinities kept.
pub fn to_writer<W: std::io::Write, T: Serialize + ?Sized>(w: W, v: &T) -> serde_json::Result<()> {
    v.serialize(Finite(&mut serde_json::Serializer::new(w)))
}

/// A serializer that hands everything to the one it wraps, the infinite floats as above.
pub struct Finite<S>(pub S);

/// A value to be written through `Finite` (the parts of a sequence, a struct, ...).
struct Wrap<'a, T: ?Sized>(&'a T);

impl<T: Serialize + ?Sized> Serialize for Wrap<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(Finite(s))
    }
}

fn f32_of(v: f32) -> f64 {
    if v.is_infinite() {
        if v > 0.0 {
            1e39
        } else {
            -1e39
        }
    } else {
        v as f64
    }
}

impl<S: Serializer> Serializer for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = Finite<S::SerializeSeq>;
    type SerializeTuple = Finite<S::SerializeTuple>;
    type SerializeTupleStruct = Finite<S::SerializeTupleStruct>;
    type SerializeTupleVariant = Finite<S::SerializeTupleVariant>;
    type SerializeMap = Finite<S::SerializeMap>;
    type SerializeStruct = Finite<S::SerializeStruct>;
    type SerializeStructVariant = Finite<S::SerializeStructVariant>;

    fn serialize_f32(self, v: f32) -> Result<S::Ok, S::Error> {
        if v.is_infinite() {
            self.0.serialize_f64(f32_of(v))
        } else {
            self.0.serialize_f32(v)
        }
    }
    fn serialize_f64(self, v: f64) -> Result<S::Ok, S::Error> {
        self.0.serialize_f64(if v.is_infinite() { f64::MAX.copysign(v) } else { v })
    }
    fn serialize_bool(self, v: bool) -> Result<S::Ok, S::Error> {
        self.0.serialize_bool(v)
    }
    fn serialize_i8(self, v: i8) -> Result<S::Ok, S::Error> {
        self.0.serialize_i8(v)
    }
    fn serialize_i16(self, v: i16) -> Result<S::Ok, S::Error> {
        self.0.serialize_i16(v)
    }
    fn serialize_i32(self, v: i32) -> Result<S::Ok, S::Error> {
        self.0.serialize_i32(v)
    }
    fn serialize_i64(self, v: i64) -> Result<S::Ok, S::Error> {
        self.0.serialize_i64(v)
    }
    fn serialize_i128(self, v: i128) -> Result<S::Ok, S::Error> {
        self.0.serialize_i128(v)
    }
    fn serialize_u8(self, v: u8) -> Result<S::Ok, S::Error> {
        self.0.serialize_u8(v)
    }
    fn serialize_u16(self, v: u16) -> Result<S::Ok, S::Error> {
        self.0.serialize_u16(v)
    }
    fn serialize_u32(self, v: u32) -> Result<S::Ok, S::Error> {
        self.0.serialize_u32(v)
    }
    fn serialize_u64(self, v: u64) -> Result<S::Ok, S::Error> {
        self.0.serialize_u64(v)
    }
    fn serialize_u128(self, v: u128) -> Result<S::Ok, S::Error> {
        self.0.serialize_u128(v)
    }
    fn serialize_char(self, v: char) -> Result<S::Ok, S::Error> {
        self.0.serialize_char(v)
    }
    fn serialize_str(self, v: &str) -> Result<S::Ok, S::Error> {
        self.0.serialize_str(v)
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<S::Ok, S::Error> {
        self.0.serialize_bytes(v)
    }
    fn serialize_none(self) -> Result<S::Ok, S::Error> {
        self.0.serialize_none()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<S::Ok, S::Error> {
        self.0.serialize_some(&Wrap(v))
    }
    fn serialize_unit(self) -> Result<S::Ok, S::Error> {
        self.0.serialize_unit()
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<S::Ok, S::Error> {
        self.0.serialize_unit_struct(name)
    }
    fn serialize_unit_variant(self, name: &'static str, index: u32, variant: &'static str) -> Result<S::Ok, S::Error> {
        self.0.serialize_unit_variant(name, index, variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, name: &'static str, v: &T) -> Result<S::Ok, S::Error> {
        self.0.serialize_newtype_struct(name, &Wrap(v))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(self, name: &'static str, index: u32, variant: &'static str, v: &T) -> Result<S::Ok, S::Error> {
        self.0.serialize_newtype_variant(name, index, variant, &Wrap(v))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, S::Error> {
        self.0.serialize_seq(len).map(Finite)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, S::Error> {
        self.0.serialize_tuple(len).map(Finite)
    }
    fn serialize_tuple_struct(self, name: &'static str, len: usize) -> Result<Self::SerializeTupleStruct, S::Error> {
        self.0.serialize_tuple_struct(name, len).map(Finite)
    }
    fn serialize_tuple_variant(self, name: &'static str, index: u32, variant: &'static str, len: usize) -> Result<Self::SerializeTupleVariant, S::Error> {
        self.0.serialize_tuple_variant(name, index, variant, len).map(Finite)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, S::Error> {
        self.0.serialize_map(len).map(Finite)
    }
    fn serialize_struct(self, name: &'static str, len: usize) -> Result<Self::SerializeStruct, S::Error> {
        self.0.serialize_struct(name, len).map(Finite)
    }
    fn serialize_struct_variant(self, name: &'static str, index: u32, variant: &'static str, len: usize) -> Result<Self::SerializeStructVariant, S::Error> {
        self.0.serialize_struct_variant(name, index, variant, len).map(Finite)
    }
    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

impl<S: ser::SerializeSeq> ser::SerializeSeq for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), S::Error> {
        self.0.serialize_element(&Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeTuple> ser::SerializeTuple for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), S::Error> {
        self.0.serialize_element(&Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeTupleStruct> ser::SerializeTupleStruct for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), S::Error> {
        self.0.serialize_field(&Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeTupleVariant> ser::SerializeTupleVariant for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), S::Error> {
        self.0.serialize_field(&Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeMap> ser::SerializeMap for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), S::Error> {
        self.0.serialize_key(&Wrap(k))
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), S::Error> {
        self.0.serialize_value(&Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeStruct> ser::SerializeStruct for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> Result<(), S::Error> {
        self.0.serialize_field(key, &Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

impl<S: ser::SerializeStructVariant> ser::SerializeStructVariant for Finite<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> Result<(), S::Error> {
        self.0.serialize_field(key, &Wrap(v))
    }
    fn end(self) -> Result<S::Ok, S::Error> {
        self.0.end()
    }
}

#[cfg(test)]
mod tests {
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct T {
        a: f32,
        b: Option<f32>,
        c: Vec<(f32, f64)>,
        d: f64,
    }

    #[test]
    fn infinities_read_back() {
        let t = T { a: f32::INFINITY, b: Some(f32::NEG_INFINITY), c: vec![(1.5, 2.25), (f32::INFINITY, -0.0)], d: 0.1 };
        let text = super::to_string(&t).expect("written");
        let back: T = super::from_str(&text).expect("read");
        assert_eq!(back, t);
        assert_eq!(super::to_string(&back).expect("again"), text);
        // (an f64 infinity comes back as the largest finite number)
        let d: f64 = super::from_str(&super::to_string(&f64::INFINITY).expect("written")).expect("read");
        assert_eq!(d, f64::MAX);
    }
}
