#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

class Fenwick {
 public:
  explicit Fenwick(std::size_t size) : values_(size + 1, 0) {}

  void Add(std::size_t index) {
    for (std::size_t cursor = index + 1; cursor < values_.size();
         cursor += cursor & -cursor) {
      ++values_[cursor];
    }
  }

  std::int64_t Prefix(std::size_t stop) const {
    std::int64_t total = 0;
    for (std::size_t cursor = stop; cursor > 0; cursor -= cursor & -cursor) {
      total += values_[cursor];
    }
    return total;
  }

 private:
  std::vector<std::int64_t> values_;
};

std::vector<std::int64_t> OrthantPerQuery(
    const double* point_x, const double* point_y, std::int64_t point_count,
    const double* query_x, const double* query_y, std::int64_t query_count,
    bool x_inclusive, bool y_inclusive) {
  std::vector<std::int64_t> point_order(point_count);
  std::vector<std::int64_t> query_order(query_count);
  std::iota(point_order.begin(), point_order.end(), 0);
  std::iota(query_order.begin(), query_order.end(), 0);
  std::stable_sort(point_order.begin(), point_order.end(), [&](auto a, auto b) {
    return point_x[a] < point_x[b];
  });
  std::stable_sort(query_order.begin(), query_order.end(), [&](auto a, auto b) {
    return query_x[a] < query_x[b];
  });

  std::vector<double> y_values(point_y, point_y + point_count);
  std::sort(y_values.begin(), y_values.end());
  y_values.erase(std::unique(y_values.begin(), y_values.end()), y_values.end());
  Fenwick tree(y_values.size());
  std::vector<std::int64_t> output(query_count, 0);
  std::int64_t cursor = 0;
  for (const auto query_index : query_order) {
    const double boundary = query_x[query_index];
    while (cursor < point_count) {
      const auto point_index = point_order[cursor];
      const bool eligible = x_inclusive ? point_x[point_index] <= boundary
                                        : point_x[point_index] < boundary;
      if (!eligible) break;
      const auto y_index = static_cast<std::size_t>(std::lower_bound(
          y_values.begin(), y_values.end(), point_y[point_index]) - y_values.begin());
      tree.Add(y_index);
      ++cursor;
    }
    const auto y_stop = static_cast<std::size_t>((y_inclusive
        ? std::upper_bound(y_values.begin(), y_values.end(), query_y[query_index])
        : std::lower_bound(y_values.begin(), y_values.end(), query_y[query_index]))
        - y_values.begin());
    output[query_index] = tree.Prefix(y_stop);
  }
  return output;
}

std::array<std::int64_t, 9> RelationMatrix(
    const double* point_x, const double* point_y, std::int64_t point_count,
    const double* query_x, const double* query_y, std::int64_t query_count) {
  const auto ll = OrthantPerQuery(point_x, point_y, point_count, query_x, query_y,
                                  query_count, false, false);
  const auto lle = OrthantPerQuery(point_x, point_y, point_count, query_x, query_y,
                                   query_count, false, true);
  const auto lel = OrthantPerQuery(point_x, point_y, point_count, query_x, query_y,
                                   query_count, true, false);
  const auto lele = OrthantPerQuery(point_x, point_y, point_count, query_x, query_y,
                                    query_count, true, true);
  std::array<std::int64_t, 9> total{};
  // One-dimensional margins are evaluated from sorted copies because the input
  // arrays preserve sample order.
  std::vector<double> sorted_x(point_x, point_x + point_count);
  std::vector<double> sorted_y(point_y, point_y + point_count);
  std::sort(sorted_x.begin(), sorted_x.end());
  std::sort(sorted_y.begin(), sorted_y.end());
  for (std::int64_t q = 0; q < query_count; ++q) {
    const auto x_less = std::lower_bound(sorted_x.begin(), sorted_x.end(), query_x[q])
                        - sorted_x.begin();
    const auto x_equal = std::upper_bound(sorted_x.begin(), sorted_x.end(), query_x[q])
                         - sorted_x.begin() - x_less;
    const auto y_less = std::lower_bound(sorted_y.begin(), sorted_y.end(), query_y[q])
                        - sorted_y.begin();
    const auto y_equal = std::upper_bound(sorted_y.begin(), sorted_y.end(), query_y[q])
                         - sorted_y.begin() - y_less;
    std::array<std::int64_t, 9> matrix{};
    matrix[0] = ll[q];
    matrix[1] = lle[q] - ll[q];
    matrix[2] = x_less - lle[q];
    matrix[3] = lel[q] - ll[q];
    matrix[4] = lele[q] - lle[q] - lel[q] + ll[q];
    matrix[5] = x_equal - matrix[3] - matrix[4];
    matrix[6] = y_less - matrix[0] - matrix[3];
    matrix[7] = y_equal - matrix[1] - matrix[4];
    matrix[8] = point_count - std::accumulate(matrix.begin(), matrix.end(), std::int64_t{0});
    for (std::size_t i = 0; i < matrix.size(); ++i) total[i] += matrix[i];
  }
  return total;
}

}  // namespace

extern "C" int task_f_transition_matrix(
    const double* id0, const double* ood0, const double* id1, const double* ood1,
    std::int64_t id_count, std::int64_t ood_count, std::int64_t* output) {
  if (!id0 || !ood0 || !id1 || !ood1 || !output || id_count <= 0 || ood_count <= 0) {
    return 1;
  }
  const auto point_relation = RelationMatrix(id0, id1, id_count, ood0, ood1, ood_count);
  // RelationMatrix is point-minus-query: less, equal, greater. These are the
  // incorrect, tie, correct states used by the Task F protocol.
  std::copy(point_relation.begin(), point_relation.end(), output);
  return 0;
}

extern "C" int task_f_query_burden(
    const double* point0, const double* point1, std::int64_t point_count,
    const double* query0, const double* query1, std::int64_t query_count,
    int point_minus_query, double* gain, double* loss, double* churn) {
  if (!point0 || !point1 || !query0 || !query1 || !gain || !loss || !churn ||
      point_count <= 0 || query_count <= 0) {
    return 1;
  }
  const auto ll = OrthantPerQuery(point0, point1, point_count, query0, query1,
                                  query_count, false, false);
  const auto lle = OrthantPerQuery(point0, point1, point_count, query0, query1,
                                   query_count, false, true);
  const auto lel = OrthantPerQuery(point0, point1, point_count, query0, query1,
                                   query_count, true, false);
  const auto lele = OrthantPerQuery(point0, point1, point_count, query0, query1,
                                    query_count, true, true);
  std::vector<double> sorted_x(point0, point0 + point_count);
  std::vector<double> sorted_y(point1, point1 + point_count);
  std::sort(sorted_x.begin(), sorted_x.end());
  std::sort(sorted_y.begin(), sorted_y.end());
  const std::array<double, 3> forward{0.0, 0.5, 1.0};
  const std::array<double, 3> reverse{1.0, 0.5, 0.0};
  const auto& utility = point_minus_query ? forward : reverse;
  for (std::int64_t q = 0; q < query_count; ++q) {
    const auto x_less = std::lower_bound(sorted_x.begin(), sorted_x.end(), query0[q])
                        - sorted_x.begin();
    const auto x_equal = std::upper_bound(sorted_x.begin(), sorted_x.end(), query0[q])
                         - sorted_x.begin() - x_less;
    const auto y_less = std::lower_bound(sorted_y.begin(), sorted_y.end(), query1[q])
                        - sorted_y.begin();
    const auto y_equal = std::upper_bound(sorted_y.begin(), sorted_y.end(), query1[q])
                         - sorted_y.begin() - y_less;
    std::array<std::int64_t, 9> matrix{};
    matrix[0] = ll[q];
    matrix[1] = lle[q] - ll[q];
    matrix[2] = x_less - lle[q];
    matrix[3] = lel[q] - ll[q];
    matrix[4] = lele[q] - lle[q] - lel[q] + ll[q];
    matrix[5] = x_equal - matrix[3] - matrix[4];
    matrix[6] = y_less - matrix[0] - matrix[3];
    matrix[7] = y_equal - matrix[1] - matrix[4];
    matrix[8] = point_count - std::accumulate(matrix.begin(), matrix.end(), std::int64_t{0});
    double g = 0.0;
    double l = 0.0;
    for (int source = 0; source < 3; ++source) {
      for (int target = 0; target < 3; ++target) {
        const double delta = utility[target] - utility[source];
        const double count = static_cast<double>(matrix[source * 3 + target]);
        if (delta > 0.0) g += delta * count;
        if (delta < 0.0) l -= delta * count;
      }
    }
    gain[q] = g;
    loss[q] = l;
    churn[q] = g + l;
  }
  return 0;
}
