#pragma once

#include "pybind11/numpy.h"

#if defined(__INTEL_COMPILER)
#  pragma warning(disable: 1682) // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#elif defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#  ifdef __clang__
//   Eigen generates a bunch of implicit-copy-constructor-is-deprecated warnings with -Wdeprecated
//   under Clang, so disable that warning here:
#    pragma GCC diagnostic ignored "-Wdeprecated"
#  endif
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wint-in-bool-context"
#  endif
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4996) // warning C4996: std::unary_negate is deprecated in C++17
#endif



#include <arrayfire.h>
#include <af/internal.h>

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

af::dtype np_dtype2af_dtype(const dtype& src_dtype) {
  const std::string dtype_str = src_dtype.attr("name").cast<std::string>();
  if (dtype_str == "float64")
    return af::dtype::f64;
  else if (dtype_str == "float32")
    return af::dtype::f32;
  // float16 is not supported in arrayfire <= 3.6.4
  // cf. https://github.com/arrayfire/arrayfire/issues/1673
  //else if(dtype_str == "float16")
  //  return af::dtype::f16;
  else if (dtype_str == "int64")
    return af::dtype::s64;
  else if (dtype_str == "uint64")
    return af::dtype::u64;
  else if (dtype_str == "int32")
    return af::dtype::s32;
  else if (dtype_str == "uint32")
    return af::dtype::u32;
  else if (dtype_str == "int16")
    return af::dtype::s16;
  else if (dtype_str == "uint16")
    return af::dtype::u16;
  // int8 is not supported in arrayfire <= 3.6.4
  // cf. https://github.com/arrayfire/arrayfire/issues/1656
  //else if (dtype_str == "int8")
  //  return af::dtype::s8;
  else if (dtype_str == "uint8")
    return af::dtype::u8;
  else if (dtype_str == "bool")
    return af::dtype::b8;
  // FIXME: not sure if they have save memory alignment for complex values
  else if (dtype_str == "complex128")
    return af::dtype::c64;
  else if (dtype_str == "complex64")
    return af::dtype::c32;
  else
    throw std::invalid_argument("unsupported numpy.dtype");
}

dtype af_dtype2np_dtype(const af::dtype& src_dtype) {
  switch(src_dtype) {
    case af::dtype::f64:
      return dtype("float64");
    case af::dtype::f32:
      return dtype("float32");
    // float16 is not supported in arrayfire <= 3.6.4
    // cf. https://github.com/arrayfire/arrayfire/issues/1673
    //case af::dtype::f16:
    //  return dtype("float16");
    case af::dtype::s64:
      return dtype("int64");
    case af::dtype::u64:
      return dtype("utint64");
    case af::dtype::s32:
      return dtype("int32");
    case af::dtype::u32:
      return dtype("uint32");
    case af::dtype::s16:
      return dtype("int16");
    case af::dtype::u16:
      return dtype("uint16");
    // int8 is not supported in arrayfire <= 3.6.4
    // cf. https://github.com/arrayfire/arrayfire/issues/1656
    //case af::dtype::s8:
    //  return dtype("int8");
    case af::dtype::u8:
      return dtype("uint8");
    case af::dtype::b8:
      return dtype("bool");
    // FIXME: not sure if they have save memory alignment for complex values
    case af::dtype::c64:
      return dtype("complex128");
    case af::dtype::c32:
      return dtype("complex64");
    default:
      throw std::invalid_argument("unsupported af::dtype");
  }
}

int af_dtype2bytes(const af::dtype& src_dtype) {
  switch(src_dtype) {
    case af::dtype::c64:
      return 16;
    case af::dtype::f64:
    case af::dtype::s64:
    case af::dtype::u64:
    case af::dtype::c32:
      return 8;
    case af::dtype::f32:
    case af::dtype::s32:
    case af::dtype::u32:
      return 4;
    case af::dtype::s16:
    case af::dtype::u16:
    //case af::dtype::f16:
      return 2;
    case af::dtype::u8:
    case af::dtype::b8:
    //case af::dtype::s8
      return 1;
    default:
      throw std::invalid_argument("unsupported af::dtype");
  }
}

array afarray2ndarray(const af::array& src) {
  // get shape
  std::vector<dim_t> shape(src.dims().ndims());
  for (int i = 0; i < src.dims().ndims(); ++i) {
    shape[i] = src.dims(i);
  }

  // get strides
  const auto bytes = af_dtype2bytes(src.type());
  const af::dim4 src_strides = af::getStrides(src);
  std::vector<dim_t> strides(src.dims().ndims());
  strides[0] = 1;
  for (int i = 0; i < src.dims().ndims() - 1; ++i)
    strides[i + 1] =  strides[i] * src.dims(i);

  for (auto& s : strides) {
    //NOTE: strides in arrayfire are pointer-based
    //      while those of numpy are byte-based
    s *= bytes;
  }

  // prepare array on host memory
  array host_data(af_dtype2np_dtype(src.type()), shape, strides);

  try {
    src.host(host_data.mutable_data());
  }
  catch (af::exception &e){
    throw std::invalid_argument("Requested type doesn't match with array");
  }

  return host_data;
}


// type_caster for af::array
template <> struct type_caster<af::array> {

  PYBIND11_TYPE_CASTER(af::array, _("af::array"));

  bool load(handle src, bool convert) {
    using StorageIndex = int;  // FIXME: correct always?
    using Index = int;
    object sparse_module = module::import("scipy.sparse");
    object spmatrix = sparse_module.attr("spmatrix");

    if (isinstance<array>(src)) { // numpy.ndarray
      // Coerce into an array, but don't do type conversion yet; the copy below handles it.
      auto buf = array::ensure(src);

      if (!buf)
          return false;

      if (buf.ndim() > 4)
        std::invalid_argument("ndim must be less than or equal to 4");

      const auto dst_af_dtype = np_dtype2af_dtype(buf.dtype());
      const auto bytes = af_dtype2bytes(dst_af_dtype);
      const bool is_f_style = (array::f_style == (buf.flags() & array::f_style));
      const bool is_c_style = (array::c_style == (buf.flags() & array::c_style));
      bool is_fortran = is_f_style & !is_c_style;
      if (!is_f_style && !is_c_style) {
        if (buf.strides(0) / bytes == 1) {
          is_fortran = true;
        }
        else if (buf.strides(buf.ndim() - 1) / bytes == 1) {
          is_fortran = false;
        }
        else {
          throw std::invalid_argument("unsupporeted strides");
        }
      }

      // cf. https://github.com/arrayfire/arrayfire/blob/70ef19897e4cf639dd720f3083dd2c6c522ff076/src/api/c/internal.cpp#L43
      af::dim4 shape(1);
      for (int i = 0; i < buf.ndim(); ++i) {
        shape[i] = buf.shape(is_fortran? i : buf.ndim() - i - 1);
      }

      af::dim4 strides(1);
      for (int i = 0; i < buf.ndim(); ++i) {
        //NOTE: strides in arrayfire are pointer-based
        //      while those of numpy are byte-based
        strides[i] = buf.strides(is_fortran? i : buf.ndim() - i - 1) / bytes;
      }

      try {
        value = af::createStridedArray(buf.data(), 0, shape, strides,
                                       dst_af_dtype, afHost);

        if (!is_fortran) {
          if (buf.ndim() <= 2)
            value = value.T();
          else if (buf.ndim() == 3)
            value = af::reorder(value, 2, 1, 0);
          else if (buf.ndim() == 4)
            value = af::reorder(value, 3, 2, 1, 0);
        }
      }
      catch (af::exception &e) {
        std::cerr << e << std::endl;
        return false;
      }
    }
    else if (isinstance(src, spmatrix)) { // scipy.sparse.base.spmatrix
      // modules
      object csr_matrix = sparse_module.attr("csr_matrix");
      object csc_matrix = sparse_module.attr("csc_matrix");
      object coo_matrix = sparse_module.attr("coo_matrix");

      object matrix_type;
      if (isinstance(src, csr_matrix))
        matrix_type = csr_matrix;
      else if (isinstance(src, csc_matrix))
        matrix_type = csc_matrix;
      else if (isinstance(src, coo_matrix))
        matrix_type = coo_matrix;
      else
        return false;

      auto obj = reinterpret_borrow<object>(src);
      if (!obj.get_type().is(matrix_type)) {
        try {
          obj = matrix_type(obj);
        } catch (const error_already_set &) {
          return false;
        }
      }

      auto values = array((object) obj.attr("data"));
      auto shape = tuple((object) obj.attr("shape"));
      const Index nnz = obj.attr("nnz").cast<Index>();
      array_t<StorageIndex> innerIndices;
      array_t<StorageIndex> outerIndices;
      af::storage storage_type;

      if (isinstance(src, csr_matrix)) {
        outerIndices = array_t<StorageIndex>((object) obj.attr("indptr"));
        innerIndices = array_t<StorageIndex>((object) obj.attr("indices"));
        storage_type = AF_STORAGE_CSR;
      }
      else if (isinstance(src, csc_matrix)) {
        outerIndices = array_t<StorageIndex>((object) obj.attr("indices"));
        innerIndices = array_t<StorageIndex>((object) obj.attr("indptr"));
        storage_type = AF_STORAGE_CSC;
      }
      else if (isinstance(src, coo_matrix)) {
        outerIndices = array_t<StorageIndex>((object) obj.attr("row"));
        innerIndices = array_t<StorageIndex>((object) obj.attr("col"));
        storage_type = AF_STORAGE_COO;
      }
      else
        return false;

      if (!values || !innerIndices || !outerIndices)
        throw std::invalid_argument("got null values or indeices");

      const auto dst_af_dtype = np_dtype2af_dtype(values.dtype());

      value = af::sparse(shape[0].cast<Index>(), shape[1].cast<Index>(),
                         nnz, values.mutable_data(),
                         outerIndices.mutable_data(), innerIndices.mutable_data(),
                         dst_af_dtype, storage_type); // FIXME: values is undefined
    }
    else { // unsupported type
      return false;
    }

    return true;
  }

  static handle cast(const af::array &src, return_value_policy /* policy */, handle /* parent */) {
    if (src.issparse()) { // sparse array
      object sparse_module = module::import("scipy.sparse");

      array innerIndices;
      array outerIndices;

      const auto src_storage_type = af::sparseGetStorage(src);
      object matrix_type;
      switch(src_storage_type) {
        case AF_STORAGE_CSR:
          matrix_type = sparse_module.attr("csr_matrix");
          innerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetRowIdx(src));
          break;
        case AF_STORAGE_CSC:
          matrix_type = sparse_module.attr("csc_matrix");
          innerIndices = afarray2ndarray(af::sparseGetRowIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          break;
        case AF_STORAGE_COO:
          matrix_type = sparse_module.attr("coo_matrix");
          innerIndices = afarray2ndarray(af::sparseGetRowIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          break;
        default:
          throw std::invalid_argument("unsupported af::dtype");
      }

      if (src_storage_type == AF_STORAGE_COO) {
        return matrix_type(
          make_tuple(afarray2ndarray(af::sparseGetValues(src)),
                     make_tuple(innerIndices, outerIndices)),
          std::make_pair(src.dims(0), src.dims(1))
        ).release();
      }
      else {
        return matrix_type(
          std::make_tuple(afarray2ndarray(af::sparseGetValues(src)),
                          innerIndices, outerIndices),
          std::make_pair(src.dims(0), src.dims(1))
        ).release();
      }
    }
    else { // dense array
      auto res = afarray2ndarray(src);
      return res.release();
    }
  }
}; // struct type_caster<af::array>


// type_caster for af::dim4
template <> struct type_caster<af::dim4> {
  PYBIND11_TYPE_CASTER(af::dim4, _("af::dim4"));

  bool load(handle src, bool convert) {
    value = af::dim4(1);

    if (isinstance<tuple>(src)) {
      const int ndims = len(src);
      if (ndims > 4)
        return false;
      tuple t = reinterpret_borrow<tuple>(src);
      for (int i = 0; i < ndims; ++i)
        value[i] = t[i].cast<dim_t>();
    }
    else if (isinstance<list>(src)) {
      const int ndims = len(src);
      if (ndims > 4)
        return false;
      list l = reinterpret_borrow<list>(src);
      for (int i = 0; i < ndims; ++i)
        value[i] = l[i].cast<dim_t>();
    }
    else {
      return false;
    }

    return true;
  }

  static handle cast(const af::dim4 &src, return_value_policy, handle) {
    switch (src.ndims()) {
      case 1:
        return make_tuple(int_((int) src[0])).release();
      case 2:
        return make_tuple(src[0], src[1]).release();
      case 3:
        return make_tuple(src[0], src[1], src[2]).release();
      case 4:
        return make_tuple(src[0], src[1], src[2], src[3]).release();
      default:
        return make_tuple().release();
    }
  }
}; // struct type_caster<af::dim4>

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)


#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
