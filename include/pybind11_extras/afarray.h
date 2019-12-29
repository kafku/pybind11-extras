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

array afarray2ndarray(const af::array& src) {
  array host_data(af_dtype2np_dtype(src.type()), src.elements());

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

      af::dim4 shape;
      for (int i = 0; i < buf.ndim(); ++i) {
        shape[i] = buf.shape(buf.ndim() - i - 1);
      }

      af_array inArray = 0;
      af_err __err = af_create_array(&inArray, buf.data(), buf.ndim(), shape.get(),
                                     np_dtype2af_dtype(buf.dtype()));
      if(__err != AF_SUCCESS)
        return false;

      value = af::array(inArray).T(); // FIXME: what if ndim == 1??
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
      std::vector<dim_t> shape(src.dims().ndims());
      for (int i = 0; i < src.dims().ndims(); ++i) {
        shape[i] = src.dims(i);
      }
      auto res = afarray2ndarray(src.T());
      res.resize(shape); // what if ndims == 1?
      return res.release();
    }
  }
}; // struct type_caster<af::array>


NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)


#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
