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

template <> struct type_caster<af::array> {
  using namespace py = pybind11;

  // FIXME: not sure if it's really appropriate
  PYBIND11_TYPE_CASTER(af::array, _<(IsRowMajor) != 0>("scipy.sparse.csr_matrix[", "scipy.sparse.csc_matrix[")
          + npy_format_descriptor<Scalar>::name + _("]"));

  bool load(handle src, bool convert) {
    py::object sparse_module = py::module::import("scipy.sparse");
    py::object spmatrix = py::module::import("scipy.sparse.base.spmatrix");
    py::object csr_matrix = sparse_module.attr("csr_matrix");
    py::object csc_matrix = sparse_module.attr("csc_matrix");
    py::object coo_matrix = sparse_module.attr("coo_matrix");

    if (isinstance<array>(src)) { // numpy.ndarray
      // If we're in no-convert mode, only load if given an array of the correct type
      if (!convert && !isinstance<array_t<Scalar>>(src))
          return false;

      // Coerce into an array, but don't do type conversion yet; the copy below handles it.
      auto buf = array::ensure(src);

      if (!buf)
          return false;

      auto dims = buf.ndim();
      if (dims < 1 || dims > 2)
          return false;

      auto fits = props::conformable(buf);
      if (!fits)
          return false;

      // Allocate the new type, then build a numpy reference into it
      value = Type(fits.rows, fits.cols);
      auto ref = reinterpret_steal<array>(eigen_ref_array<props>(value));
      if (dims == 1) ref = ref.squeeze();
      else if (ref.ndim() == 1) buf = buf.squeeze();

      int result = detail::npy_api::get().PyArray_CopyInto_(ref.ptr(), buf.ptr());

      if (result < 0) { // Copy failed!
          PyErr_Clear();
          return false;
      }

      switch (dims) {
      case 1:
        value = af::array(buf.size(), buf.mutable_data());
      case 2:
        // NOTE: arrayfire assume its arrays to be colum-major, so we need to transpose
        value = af::array(buf.shape(1), buf.shape(0), buf.mutable_data()).T();
      case 3:
        value = af::array(buf.shape(2), buf.shape(1), buf.shape(0),
                          buf.mutable_data()).T();
      case 4:
        value = af::array(buf.shape(3), buf.shape(2), buf.shape(1), buf.shape(0),
                          buff.mutable_data()).T();
      default:
        return false;
      }
    }
    else if (isinstance(src, spmatrix)) { // scipy.sparse.base.spmatrix
      using StorageIndex = int;  // FIXME: correct??
      using Index = int;

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

      py::dtype src_dtype = obj.attr("dtype");   // FIXME: work?
      const auto dst_af_dtype = np_dtype2af_dtype(src_dtype);

      //auto values = py::array_t<Scalar>((py::object) obj.attr("data"));
      auto values = py::array((py::object) obj.attr("data"));  // FIXME: not sure if this work
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

  // Normal returned non-reference, non-const value:
  static handle cast(Type &&src, return_value_policy /* policy */, handle parent) {
    return cast_impl(&src, return_value_policy::move, parent);
  }
  // If you return a non-reference const, we mark the numpy array readonly:
  static handle cast(const Type &&src, return_value_policy /* policy */, handle parent) {
    return cast_impl(&src, return_value_policy::move, parent);
  }
  // lvalue reference return; default (automatic) becomes copy
  static handle cast(Type &src, return_value_policy policy, handle parent) {
    if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
      policy = return_value_policy::copy;
    return cast_impl(&src, policy, parent);
  }
  // const lvalue reference return; default (automatic) becomes copy
  static handle cast(const Type &src, return_value_policy policy, handle parent) {
    if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
      policy = return_value_policy::copy;
    return cast(&src, policy, parent);
  }
  // non-const pointer return
  static handle cast(Type *src, return_value_policy policy, handle parent) {
      return cast_impl(src, policy, parent);
  }
  // const pointer return
  static handle cast(const Type *src, return_value_policy policy, handle parent) {
      return cast_impl(src, policy, parent);
  }

private:
  af::dtype np_dtype2af_dtype(py::dtype src_dtype) {
    switch() { // FIXME: how to get dtype.name?
      case "float64":
        return af::dtype::f64;
      case "float32":
        return af::dtype::f64;
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
}; // struct type_caster<af::array>


// sparse array
template <af::dtype AF_DTYPE, af::storage AF_STORAGETYPE>
struct type_caster<typed_array<AF_DTYPE, AF_STORAGETYPE>> {
  using Typte = typed_array<AF_DTYPE, AF_STORAGETYPE>;
  using Scalar = typename pybaf::traits::dtype2cpp<AF_DTYPE>::type; // FIXME: what if complex float???
  static constexpr bool rowMajor = (AF_STORAGETYPE == AF_STORAGE_CSR);
  static constexpr bool colMajor = (AF_STORAGETYPE == AF_STORAGE_CSC);

  bool load(handle src, bool convert) {
    auto obj = reinterpret_borrow<object>(src);
    object sparse_module = module::import("scipy.sparse");
    object matrix_type = sparse_module.attr(
        rowMajor ? "csr_matrix" : (colMajor ?  "csc_matrix" : "coo_matrix"));

    if (!obj.get_type().is(matrix_type)) {
        try {
            obj = matrix_type(obj);
        } catch (const error_already_set &) {
            return false;
        }
    }

    auto values = array_t<Scalar>((object) obj.attr("data"));
    auto row_indices = array_t<int>((object) obj.attr(               // FIXME: int?
          pybaf::traits::af_storage_traits<AF_STORAGETYPE>::row_idx));
    auto col_indices = array_t<int>((object) obj.attr(               // FIXME: int?
          pybaf::traits::af_storage_traits<AF_STORAGETYPE>::col_idx));
    auto shape = pybind11::tuple((pybind11::object) obj.attr("shape"));
    auto nnz = obj.attr("nnz").cast<dim_t>();

    if (!values || !row_indices || !col_indices)
        return false;

    value = af::sparse(
        shape[0].cast<dim_t>(), shape[1].cast<dim_t>(), nnz,
        values.mutable_data(), row_indices.mutable_data(), col_indicesmutable_data(),  // FIXME: need to be checked
        AF_DTYPE, AF_STORAGETYPE);

    return true;
  }

  // TODO:
  static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
      const_cast<Type&>(src).makeCompressed();

      object matrix_type = module::import("scipy.sparse").attr(
          rowMajor ? "csr_matrix" : "csc_matrix");  // FIXME: what if COO?

      array data(src.nonZeros(), src.valuePtr());
      array outerIndices((rowMajor ? src.rows() : src.cols()) + 1, src.outerIndexPtr());
      array innerIndices(src.nonZeros(), src.innerIndexPtr());

      return matrix_type(
          std::make_tuple(data, innerIndices, outerIndices),
          std::make_pair(src.rows(), src.cols())
      ).release();
  }

  // FIXME
  PYBIND11_TYPE_CASTER(Type, _<(IsRowMajor) != 0>("scipy.sparse.csr_matrix[", "scipy.sparse.csc_matrix[")
          + npy_format_descriptor<Scalar>::name + _("]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)


#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
