from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# ── DATASET NOTE ─────────────────────────────────────────────────────────────
# locations_data.csv relevant columns:
#   latitude, longitude  → passed as locations list
#   demand_units         → passed as demands list (0 for depot/warehouse rows)
#   location_type        → filter to 'warehouse' for depot selection
#
# Example usage:
#   df = pd.read_csv('data/locations_data.csv')
#   city_df = df[df['city'] == 'Mumbai'].head(21)  # 1 depot + 20 stops
#   depot   = city_df[city_df['location_type'] == 'warehouse'].index[0]
#   locs    = city_df[['latitude','longitude']].values.tolist()
#   demands = city_df['demand_units'].tolist()

_ROUTING_STATUS = {
    0: 'ROUTING_NOT_SOLVED',
    1: 'ROUTING_SUCCESS',
    2: 'ROUTING_FAIL',
    3: 'ROUTING_FAIL_TIMEOUT',
    4: 'ROUTING_INVALID',
    5: 'ROUTING_INFEASIBLE',
}


class RouteOptimizer:

    def __init__(self, num_vehicles: int = 3, max_distance_km: int = 500,
                 time_limit_seconds: int = 10, depot_index: int = 0):
        self.num_vehicles       = num_vehicles
        self.max_distance_km    = max_distance_km
        self.time_limit_seconds = time_limit_seconds
        self.depot_index        = depot_index

    def create_distance_matrix(self, locations: list) -> list:
        """Haversine distance matrix (km, rounded to nearest int)."""
        coords = np.radians(np.array(locations, dtype=float))
        lat = coords[:, 0]
        lon = coords[:, 1]

        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]

        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))

        matrix = np.round(6371 * c).astype(int)
        np.fill_diagonal(matrix, 0)
        return matrix.tolist()

    def solve(self, locations: list,
              demands: list = None,
              vehicle_capacity: int = None) -> dict:
        """
        Solve VRP / CVRP.

        Parameters
        ----------
        locations        : [[lat, lon], ...]  — first entry is the depot.
        demands          : demand per location (depot = 0). Enables CVRP when
                           combined with vehicle_capacity.
        vehicle_capacity : max load per vehicle (same units as demands).
        """
        distance_matrix = self.create_distance_matrix(locations)

        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix), self.num_vehicles, self.depot_index
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            return distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
        routing.AddDimension(transit_cb_idx, 0, self.max_distance_km, True, 'Distance')

        if demands is not None and vehicle_capacity is not None:
            def demand_callback(from_idx):
                return demands[manager.IndexToNode(from_idx)]

            demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_cb_idx,
                0,
                [vehicle_capacity] * self.num_vehicles,
                True,
                'Capacity',
            )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.time_limit_seconds

        solution = routing.SolveWithParameters(search_params)

        if not solution:
            status_code = routing.status()
            return {
                'status':        'No solution found',
                'solver_status': _ROUTING_STATUS.get(status_code, str(status_code)),
            }

        routes         = []
        total_distance = 0

        for v in range(self.num_vehicles):
            idx   = routing.Start(v)
            route = []
            dist  = 0

            while not routing.IsEnd(idx):
                route.append(manager.IndexToNode(idx))
                prev_idx = idx
                idx  = solution.Value(routing.NextVar(idx))
                dist += routing.GetArcCostForVehicle(prev_idx, idx, v)

            route.append(manager.IndexToNode(idx))

            if len(route) > 2 or route[0] != route[-1] or dist > 0:
                routes.append({
                    'vehicle':     v,
                    'route':       route,
                    'distance_km': dist,
                })
            total_distance += dist

        return {
            'status':            'success',
            'routes':            routes,
            'vehicles_used':     len(routes),
            'total_distance_km': total_distance,
        }
