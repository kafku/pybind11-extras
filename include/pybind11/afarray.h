#pragma once

#include "numpy.h"

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
  switch(src_dtype.attr("name").cast<std::string>()) {
    case "float64":
      return af::dtype::f64;
    case "float32":
      return af::dtype::f32;
    // float16 is not supported in arrayfire <= 3.6.4
    // cf. https://github.com/arrayfire/arrayfire/issues/1673
    //case "float16":
    //  return af::dtype::f16;
    case "int64":
      return af::dtype::s64;
    case "uint64":
      return af::dtype::u64;
    case "int32":
      return af::dtype::s32;
    case "uint32":
      return af::dtype::u32;
    case "int16":
      return af::dtype::s16;
    case "uint16":
      return af::dtype::u16;
    // int8 is not supported in arrayfire <= 3.6.4
    // cf. https://github.com/arrayfire/arrayfire/issues/1656
    //case "int8":
    //  return af::dtype::s8;
    case "uint8":
      return af::dtype::u8;
    case "bool":
      return af::dtype::b8;
    // FIXME: not sure if they have save memory alignment for complex values
    case "complex128":
      return af::dtype::c64;
    case "complex64":
      return af::dtype::c32;
    default:
      throw std::invalid_argument("unsupported numpy.dtype");
  }
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

template <typename T>
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
  namespace py = pybind11;

  PYBIND11_TYPE_CASTER(af::array, _("af::array"));

  bool load(handle src, bool convert) {
    using StorageIndex = int;  // FIXME: correct always?
    using Index = int;
    py::object sparse_module = py::module::import("scipy.sparse");
    py::object spmatrix = py::module::import("scipy.sparse.base.spmatrix");

    if (isinstance<py::array>(src)) { // numpy.ndarray
      // Coerce into an array, but don't do type conversion yet; the copy below handles it.
      auto buf = py::array::ensure(src);

      if (!buf)
          return false;

      af::dim4 shape;
      for (int i = 0; i < buf.ndim(); ++i) {
        shape[i] = buf.dims(buf.ndim() - i - 1);
      }

      af_array inArray = 0;
      af_err __err = af_create_array(&inArray, buf.data(), buf.ndim(), shape.get(),
                                     np_dtype2af_dtype(src));
      if(__err != AF_SUCCESS)
        return false;

      value = af::array(inArray).T(); // FIXME: what if ndim == 1??
    }
    else if (isinstance(src, spmatrix)) { // scipy.sparse.base.spmatrix
      // modules
      py::object csr_matrix = sparse_module.attr("csr_matrix");
      py::object csc_matrix = sparse_module.attr("csc_matrix");
      py::object coo_matrix = sparse_module.attr("coo_matrix");

      py::object matrix_type;
      if (isinstance(src, csr_matrix))
        matrix_type = csr_matrix;
      else if (isinstance(src, csc_matrix))
        matrix_type = csc_matrix;
      else if (isinstance(src, coo_matrix))
        matrix_type = coo_matrix;
      else
        return false;

      auto obj = py::reinterpret_borrow<py::object>(src);
      if (!obj.get_type().is(matrix_type)) {
        try {
          obj = matrix_type(obj);
        } catch (const error_already_set &) {
          return false;
        }
      }

      auto values = py::array((py::object) obj.attr("data"));
      auto shape = py::tuple((py::object) obj.attr("shape"));
      const Index nnz = obj.attr("nnz").cast<Index>();
      py::array_t<StorageIndex> innerIndices;
      py::array_t<StorageIndex> outerIndices;
      af::storage storage_type;

      if (isinstance(src, csr_matrix)) {
        outerIndices = py::array_t<StorageIndex>((py::object) obj.attr("indptr"));
        innerIndices = py::array_t<StorageIndex>((py::object) obj.attr("indices"));
        storage_type = AF_STORAGE_CSR;
      }
      else if (isinstance(src, csc_matrix)) {
        outerIndices = py::array_t<StorageIndex>((py::object) obj.attr("indices"));
        innerIndices = py::array_t<StorageIndex>((py::object) obj.attr("indptr"));
        storage_type = AF_STORAGE_CSC;
      }
      else if (isinstance(src, coo_matrix)) {
        outerIndices = py::array_t<StorageIndex>((py::object) obj.attr("row"));
        innerIndices = py::array_t<StorageIndex>((py::object) obj.attr("col"));
        storage_type = AF_STORAGE_COO;
      }

      if (!values || !innerIndices || !outerIndices)
        throw std::invalid_argument("got null values or indeices");

      const auto dst_af_dtype = np_dtype2af_dtype(obj.attr("dtype"));

      value = af::sparse(shape[0].cast<Index>(), shape[1].cast<Index>(),
                         nnz, valuse.mutable_data(),
                         outerIndices.mutable_data(), innerIndices.mutable_data(),
                         dst_af_dtype, storage_type);
    }
    else { // unsupported type
      return false;
    }

    return true;
  }

  static handle cast(const af::array &src, return_value_policy /* policy */, handle /* parent */) {
    if (src.issparse()) { // sparse array
      object sparse_module = py::module::import("scipy.sparse");

      array innerIndices;
      array outerIndices;

      const auto src_storage_type = af::sparseGetStorage(src);
      object matrix_type;
      switch(src_storage_type) {
        case AF_STORAGE_CSR:
          matrix_type = sparse_module.attr("csr_matrix");
          innerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetRowIdx(src)); // FIXME: check
          break;
        case AF_STORAGE_CSC:
          matrix_type = sparse_module.attr("csc_matrix");
          innerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetRowIdx(src)); // FIXME: check
          break;
        case AF_STORAGE_COO:
          matrix_type = sparse_module.attr("coo_matrix");
          innerIndices = afarray2ndarray(af::sparseGetRowIdx(src));
          outerIndices = afarray2ndarray(af::sparseGetColIdx(src));
          break;
        default:
          throw std::invalid_argument("unsupported af::dtype");
      }

      return matrix_type(
        std::make_tuple(afarray2ndarray(af::sparseGetValues(src)),
                        innerIndices, outerIndices),
        std::make_pair(src.dims(0), src.dism(1))
      ).release();
    }
    else { // dense array
      std::vector<dim_t> shape(src.ndims());
      for (int i = 0; i < src.ndims(); ++i) {
        shape[i] = src.dims(i);
      }
      return afarray2ndarray(src.T()).resize(shape).release();
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
