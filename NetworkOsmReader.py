import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
from pathlib import Path
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from shapely.geometry import Point, LineString, Polygon
from math import sin, cos, sqrt, atan2, radians
from itertools import islice
import networkx as nx
import re
import copy
from functools import cmp_to_key
import cmath
import os
import glob


def distance_between_points(p, q):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(p[0])
    lon1 = radians(p[1])
    lat2 = radians(q[0])
    lon2 = radians(q[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    # in km
    return distance


def road_graph_to_edge_gpd(road_graph):
    """
    store road segments into a geppandas dataframe
    :param road_graph: a road network graph in networkx
    :return gpd_edges: a geopandas dataframe of road segments
    """
    gpd_edges = gpd.GeoDataFrame(columns=('from', 'to', 'geometry', 'length', 'highway'))
    for e_from, e_to, data in road_graph.edges(data=True):
        if 'geometry' in data:
            s = gpd.GeoSeries({'from': e_from,
                               'to': e_to,
                               'geometry': data['geometry'],
                               'length': data['length'],
                               'highway': data['highway']})
            gpd_edges = gpd_edges.append(s, ignore_index=True)
        else:
            p1 = Point(road_graph.nodes[e_from]['x'], road_graph.nodes[e_from]['y'])
            p2 = Point(road_graph.nodes[e_to]['x'], road_graph.nodes[e_to]['y'])
            data.update({'geometry': LineString((p1, p2))})
            s = gpd.GeoSeries({'from': e_from,
                               'to': e_to,
                               'geometry': LineString((p1, p2)),
                               'length': data['length'],
                               'highway': data['highway']})
            gpd_edges = gpd_edges.append(s, ignore_index=True)
    gpd_edges.crs = road_graph.graph['crs']
    gpd_edges.name = 'edges'
    gpd_edges['bbox'] = gpd_edges.apply(lambda row: row['geometry'].bounds, axis=1)
    return gpd_edges


def get_max_speed(highway):
    """
    return the corresponding max speed in kmph
    """
    if highway == 'mortorway':
        return 100
    elif highway == 'mortorway_link':
        return 60
    elif highway == 'trunk':
        return 80
    elif highway == 'trunk_link':
        return 40
    elif highway == 'primary':
        return 60
    elif highway == 'primary_link':
        return 40
    elif highway == 'secondary':
        return 50
    elif highway == 'secondary_link':
        return 20
    elif highway == 'residential':
        return 30
    elif highway == 'teritiary':
        return 50
    elif highway == 'teritiary_link':
        return 20
    elif highway == 'living_street':
        return 20
    elif highway == 'road':
        return 20
    elif highway == 'service':
        return 20
    else:
        return 50


def get_max_speeds(gpd_edges_utm):
    max_speeds = []
    for idx, row in gpd_edges_utm.iterrows():
        if isinstance(row['highway'], list):
            max_speed1 = get_max_speed(row['highway'][0])
            max_speed2 = get_max_speed(row['highway'][1])
            if row['length'] > 100:
                max_speed = max(max_speed1, max_speed2)
                max_speeds.append(max_speed)
            else:
                max_speed = min(max_speed1, max_speed2)
                max_speeds.append(max_speed)
        else:
            max_speeds.append(get_max_speed(row['highway']))
    return max_speeds


def build_rtree_index_edges(gpd_edges):
    """
    build a r-tree index for road segments
    input:
        gpd_edges: a geopandas dataframe that contains road segments (edge geometries)
    output:
        idx: a r-tree index of the edge geometries
    """
    # r-tree index for edges
    from rtree import index
    p = index.Property()
    idx = index.Index(properties=p)
    for index, row in gpd_edges.iterrows():
        idx.insert(index, row['bbox'], obj=row)
    return idx


def read_trip(filename):
    """
    read trajectory from csv file
    """
    # col_names = ['obj_id', 'lat', 'lon', 'timestamp', 'unknown1', 'unknow2']
    col_names = ['obj_id', 'timestamp', 'lat', 'lon', 'uk1', 'uk2', 'uk3']
    trip = pd.read_csv(filename, header=None, names=col_names)
    trip.drop(['uk1', 'uk2', 'uk3'], axis=1, inplace=True)
    trip['timestamp'] = pd.to_datetime(trip['timestamp'], format="%Y-%m-%d %H:%M:%S")

    print('Init trip length: ' + str(len(trip)))
    dropIndices = []
    for i in range(len(trip)):
        first = trip.iloc[i]
        second = trip.iloc[i + 1]
        dist = distance_between_points((first['lat'], first['lon']), (second['lat'], second['lon']))
        timeDelta = second['timestamp'] - first['timestamp']
        secondDiff = timeDelta.total_seconds()
        if dist <= 0.1 and secondDiff < 60:
            dropIndices.append(i)
        i = i + 2
        if i > len(trip) - 1:
            break

    trip = trip.drop(trip.index[dropIndices])

    print('Stripped trip length: ' + str(len(trip)))
    trip['geometry'] = trip.apply(lambda z: Point(z.lon, z.lat), axis=1)
    trip = gpd.GeoDataFrame(trip)
    return trip


def transform_coordinates(point, crs, to_crs):
    return ox.project_geometry(point, crs, to_crs)[0]


def query_k_nearest_road_segments(edge_idx, point, k):
    """
    query k-nearest road segments of a given point
    :param edge_idx: the road segments r-tree index
    :param point: the given point
    :param k: the number of segments needed to query
    :return: k candidates as a pandas DataFrame
    """
    from shapely.ops import nearest_points
    candidates = pd.DataFrame(columns=('distance', 'from', 'to', 'proj_point', 'road'))
    hits = edge_idx.nearest((point.x, point.y, point.x, point.y), k, objects=True)
    # print(hits)
    for item in hits:
        # print(item)
        # print(item.object)
        results = nearest_points(point, item.object['geometry'])
        # print('result count: ' + str(len(results)))
        d = point.distance(results[1])
        # if d > 50:
        #     continue
        s = pd.Series({'distance': d,
                       'from': item.object['from'],
                       'to': item.object['to'],
                       'proj_point': results[1],
                       'road': item.object})
        candidates = candidates.append(s, ignore_index=True)
    candidates.sort_values(by='distance', axis=0, inplace=True)
    return candidates


def find_candidates(trip, edge_idx, k):
    """
    given a trip, find candidates points for each point in the trip
    :param trip: a GPS trajectory (without coordinates transform)
    :param edge_idx: road segments r-tree index of the corresponding road network
    :param k: the number of candidates
    :return: the trip with candidates
    """
    candi_list = []
    for i in range(0, len(trip)):
        # print('FINDING CANDIDATES FOR I = ' + str(i))
        candidates = query_k_nearest_road_segments(edge_idx, trip.iloc[i]['geometry_utm'], k * 3)
        # print('length of candidate sub list: ' + str(len(candidates)))
        distanceFilter = candidates['distance'] <= 50
        candidates = candidates[distanceFilter]
        # print('length of candidates after filtering: ' + str(len(candidates)))
        candi_list.append(candidates[0:min(k, len(candidates))])
    # print('length of candidate list: ' + str(len(candi_list)))
    trip['candidates'] = candi_list


def trip_bbox(trip):
    """
    get the bounding box of the given trip
    input: trip: a trajectory
    output: (minx, miny, maxx, maxy)
    """
    from shapely.geometry import LineString
    line = LineString(list(trip['geometry']))
    return line.bounds


def k_shortest_paths(G, source, target, k, weight=None):
    try:
        convertedGraph = nx.DiGraph(G)
        return list(islice(nx.shortest_simple_paths(convertedGraph, source, target, weight=weight), k))
    except nx.NetworkXNoPath:
        # print('K-SHORTEST-PATHS ERROR: NO PATH BETWEEN SOURCE/TARGET NODES')
        # print('SOURCE NODE ID: ' + str(source))
        # print('TARGET NODE ID: ' + str(target))
        return []
    except nx.NetworkXError:
        # print('K-SHORTEST-PATHS ERROR: SOURCE/TARGET NODES NOT IN THE GRAPH')
        return []
    except nx.NetworkXNotImplemented:
        # print('K-SHORTEST-PATHS ERROR: GRAPH IS MULTI[DI]GRAPH')
        return []
    except:
        # print('OTHER K-SHORTEST-PATHS EXCEPTION')
        return []


def node_list_to_path(G, node_list):
    """
    Given a list of nodes, return a list of lines that together
    follow the path
    defined by the list of nodes.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start),
    (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        edge_data = G.get_edge_data(u, v)
        if edge_data is not None:
            # if there are parallel edges, select the shortest in length
            data = min(G.get_edge_data(u, v).values(),
                       key=lambda x: x['length'])
            # if it has a geometry attribute
            if 'geometry' in data:
                # add them to the list of lines to plot
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute,p
            # then the edge is a straight line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    return lines


def plot_path(lats, longs):
    base_dir = Path(__file__).parent.resolve()
    for j in range(len(lats)):
        for i in range(len(lats[j])):
            polyline = []
            color = 'blue' if j % 2 else 'yellow'
            for k in range(len(lats[j][i])):
                tup = (lats[j][i][k], longs[j][i][k])
                polyline.append(tup)
            if len(polyline) > 1:
                foliumMap = folium.Map(location=[lats[j][i][0], longs[j][i][0]], zoom_start=15)
                folium.PolyLine(polyline, color=color, weight=2.5, opacity=1).add_to(foliumMap)
                # filePath = 'foliumMaps/foliumMapPaths' + str(j) + '_' + str(i) + '.html'
                # foliumMap.save(str(base_dir / filePath))


def plot_full_paths(lats, longs):
    base_dir = Path(__file__).parent.resolve()
    for i in range(len(lats)):
        polyline = []
        color = 'blue' if i % 2 else 'yellow'
        for k in range(len(lats[i])):
            tup = (lats[i][k], longs[i][k])
            polyline.append(tup)
        if len(polyline) > 1:
            foliumMap = folium.Map(location=[lats[i][0], longs[i][0]], zoom_start=15)
            folium.PolyLine(polyline, color=color, weight=2.5, opacity=1).add_to(foliumMap)
            # filePath = 'foliumMaps/foliumMapPaths' + str(i) + '.html'
            # foliumMap.save(str(base_dir / filePath))


def plot_full_paths_numbered(lats, longs, j, path, road_graph):
    base_dir = Path(__file__).parent.resolve()
    for i in range(len(lats)):
        polyline = []
        color = 'blue' if i % 2 else 'black'
        for k in range(len(lats[i])):
            tup = (lats[i][k], longs[i][k])
            polyline.append(tup)
        if len(polyline) > 1:
            foliumMap = folium.Map(location=[lats[i][0], longs[i][0]], zoom_start=15)
            folium.PolyLine(polyline, color=color, weight=2.5, opacity=1).add_to(foliumMap)

            for node in path[0]:
                # print(node)
                nodeData = road_graph.nodes[node]
                folium.Marker((nodeData['y'], nodeData['x']), icon=folium.Icon(color='red')).add_to(foliumMap)
                tooltip = 'Click me!'
                folium.Marker((nodeData['y'], nodeData['x']), icon=folium.Icon(color='green'),
                              popup='<i>Candidate: ' + str(nodeData['osmid']) + '</i>',
                              tooltip=tooltip).add_to(foliumMap)

            filePath = 'foliumMaps/foliumMapPaths_' + str(j) + '_' + str(i) + '.html'
            foliumMap.save(str(base_dir / filePath))


def plot_full_paths_numbered2(lats, longs, j, path, road_graph):
    base_dir = Path(__file__).parent.resolve()
    for i in range(len(lats)):
        polyline = []
        color = 'blue' if i % 2 else 'black'
        for k in range(len(lats[i])):
            tup = (lats[i][k], longs[i][k])
            polyline.append(tup)
        if len(polyline) > 1:
            foliumMap = folium.Map(location=[lats[i][0], longs[i][0]], zoom_start=15)
            folium.PolyLine(polyline, color=color, weight=2.5, opacity=1).add_to(foliumMap)

            for node in path:
                # print(node)
                nodeData = road_graph.nodes[node]
                folium.Marker((nodeData['y'], nodeData['x']), icon=folium.Icon(color='red')).add_to(
                    foliumMap)
                tooltip = 'Click me!'
                folium.Marker((nodeData['y'], nodeData['x']), icon=folium.Icon(color='green'),
                              popup='<i>Candidate: ' + str(nodeData['osmid']) + '</i>',
                              tooltip=tooltip).add_to(foliumMap)

            filePath = 'foliumMaps/foliumMapPaths_' + str(j) + '_' + str(i) + '.html'
            foliumMap.save(str(base_dir / filePath))


def getAccelerationDistance(speedDelta, currentSpeed):
    defaultMaxAcceleration = 4.5
    accelerationTime = abs(speedDelta / defaultMaxAcceleration)
    accelerationDistance = abs((0.5 * defaultMaxAcceleration * (accelerationTime * accelerationTime)) + \
                               (currentSpeed * accelerationTime))
    return accelerationDistance


def extractDebugInformationOfNodeList(G, node_list, timeDelta, firstCandidate, secondCandidate, crs, to_crs):
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    maxspeeds = []
    lengths = []
    time_requireds = []
    total_time = 0.0
    for u, v in edge_nodes:
        edge_data = G.get_edge_data(u, v)
        if edge_data is not None:
            if 'maxspeed' in edge_data[0]:
                if len(re.findall(r'\d+', str(edge_data[0]['maxspeed']))) > 0:
                    maxspeed = float(re.findall(r'\d+', str(edge_data[0]['maxspeed']))[0])
                    maxspeed = (maxspeed * 1.609344) / 3.6
                else:
                    maxspeed = (70 * 1.609344) / 3.6
            else:
                maxspeed = (70 * 1.609344) / 3.6

            if u == firstCandidate['from'] and v == firstCandidate['to']:
                length = edge_data[0]['length']
                # print('segment length: ' + str(length))
                p = transform_coordinates(firstCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                uNode = G.nodes[u]
                d = distance_between_points((uNode['y'], uNode['x']), projectionPoint)
                d = d * 1000
                # print('distance from proj_point to u =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            elif u == secondCandidate['from'] and v == secondCandidate['to']:
                length = edge_data[0]['length']
                # print('segment length: ' + str(length))
                p = transform_coordinates(secondCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                vNode = G.nodes[v]
                d = distance_between_points((vNode['y'], vNode['x']), projectionPoint)
                d = d * 1000
                # print('distance from proj_point to v =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            else:
                length = edge_data[0]['length']

            time_required_to_travel_the_road = length / maxspeed
            maxspeeds.append(maxspeed)
            lengths.append(length)
        else:
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            if u == firstCandidate['from'] and v == firstCandidate['to']:
                length = distance_between_points(line[0], line[1])
                length = length * 1000
                # print('segment length: ' + str(length))
                p = transform_coordinates(firstCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                uNode = G.nodes[u]
                d = distance_between_points((uNode['y'], uNode['x']), projectionPoint)
                d = d * 1000
                # print('distance from proj_point to u =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            elif u == secondCandidate['from'] and v == secondCandidate['to']:
                length = distance_between_points(line[0], line[1])
                length = length * 1000
                # print('segment length: ' + str(length))
                p = transform_coordinates(secondCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                vNode = G.nodes[v]
                d = distance_between_points((vNode['y'], vNode['x']), projectionPoint)
                d = d * 1000
                # print('distance from proj_point to v =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            else:
                length = distance_between_points(line[0], line[1])
                length = length * 1000
            maxspeed = (70 * 1.609344) / 3.6
            time_required_to_travel_the_road = length / maxspeed
            maxspeeds.append(str(maxspeed) + ' - d')
            lengths.append(str(length) + ' - d')
        total_time += time_required_to_travel_the_road
        time_requireds.append(time_required_to_travel_the_road)

    total_time_acceleration = node_list_to_path_time_required(G, node_list, firstCandidate, secondCandidate,
                                                              crs, to_crs)

    debugInformationForEdge = {
        'maxspeed': maxspeeds,
        'length': lengths,
        'time_required_naive': time_requireds,
        'total_time': total_time,
        'total_time_acceleration': total_time_acceleration
    }

    return debugInformationForEdge


def node_list_to_path_time_required(G, node_list):
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    total_path_time_required = 0.0
    total_path_time_required_acceleration = 0.0

    distanceSpeedTuples = []

    for u, v in edge_nodes:
        edge_data = G.get_edge_data(u, v)
        if edge_data is not None:
            if 'maxspeed' in edge_data[0]:
                if len(re.findall(r'\d+', str(edge_data[0]['maxspeed']))) > 0:
                    maxspeed = float(re.findall(r'\d+', str(edge_data[0]['maxspeed']))[0])
                    # print('FOUND IN OSM MAXSPEED: ' + str(maxspeed))
                    maxspeed = (maxspeed * 1.609344) / 3.6
                else:
                    maxspeed = (70 * 1.609344) / 3.6
            else:
                maxspeed = (70 * 1.609344) / 3.6
            length = edge_data[0]['length']
            # print('MAXSPEED = ' + str(maxspeed))
            time_required_to_travel_the_road = length / maxspeed
            total_path_time_required += time_required_to_travel_the_road
            distanceSpeedTuples.append((length, maxspeed))
        else:
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            length = distance_between_points(line[0], line[1])
            maxspeed = (70 * 1.609344) / 3.6
            time_required_to_travel_the_road = length / maxspeed
            total_path_time_required += time_required_to_travel_the_road
            # print('MAXSPEED = ' + str(maxspeed))
            distanceSpeedTuples.append((length, maxspeed))

    prevSpeed = 0.0

    # print('looking at a path with #' + str(len(distanceSpeedTuples)) + ' distance tuples')
    # print('length of node list: ' + str(len(node_list)))

    # default acceleration / deceleration in m/s^2
    defaultMaxAcceleration = 4.5
    for i in range(0, len(distanceSpeedTuples)):
        # print('i: ' + str(i))

        nextSpeed = distanceSpeedTuples[i + 1][1]
        currentSpeed = distanceSpeedTuples[i][1]
        currentDistance = distanceSpeedTuples[i][0]

        if i + 1 == len(distanceSpeedTuples):
            # print('driving the last segment:')
            # timeRequired = distanceSpeedTuples[i][0] / distanceSpeedTuples[i][1]
            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(distanceSpeedTuples[i][0]))
            # print('default timeRequired = ' + str(distanceSpeedTuples[i][0] / distanceSpeedTuples[i][1]))
            # total_path_time_required_acceleration += timeRequired

            # TODO: decelerate to 0 mph
            speedDeltaNextSegment = nextSpeed - currentSpeed
            decelerationDistance = getAccelerationDistance(speedDeltaNextSegment, currentSpeed)
            if decelerationDistance > currentDistance:
                a = 0.5 * defaultMaxAcceleration
                b = currentSpeed
                c = -1.0 * currentDistance
                d = (b ** 2) - (4 * a * c)
                sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                if type(sol1) == complex:
                    sol1 = sol1.real
                if type(sol2) == complex:
                    sol2 = sol2.real
                if sol1 <= 0:
                    sol = sol2
                elif sol2 <= 0:
                    sol = sol1
                elif sol1 < sol2:
                    sol = sol1
                else:
                    sol = sol2
                newSpeed = currentSpeed + (-1.0 * defaultMaxAcceleration * sol)
                prevSpeed = newSpeed
                timeRequired = sol
            else:
                restDistance = currentDistance - decelerationDistance
                restTime = restDistance / currentSpeed
                prevSpeed = nextSpeed
                decelerationTime = abs(speedDeltaNextSegment / defaultMaxAcceleration)
                timeRequired = decelerationTime + restTime
            total_path_time_required_acceleration += timeRequired
            break
            # TODO: decelerate to 0 mph END

        if prevSpeed == 0.0:
            prevSpeed = currentSpeed

            # TODO: init speed to 0 mph
            prevSpeed = 0.0
            # TODO: init speed to 0 mph END

        timeRequired = 0.0

        # speed diff between previous segment and current segment
        speedDeltaThisSegment = currentSpeed - prevSpeed
        if speedDeltaThisSegment == 0.0:
            # print('speed up not required, prevSpeed = ' + str(prevSpeed) + ' and currentSpeed = ' + str(currentSpeed))
            # current speed reached, figure out what to do to compensate for next segment if required
            speedDeltaNextSegment = nextSpeed - currentSpeed

            if speedDeltaNextSegment >= 0.0:
                # print('slow down not required, nextSpeed = ' + str(nextSpeed))
                # speed is fine, figure out time required to travel segment
                timeRequired = currentDistance / currentSpeed
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                prevSpeed = currentSpeed

            elif speedDeltaNextSegment < 0.0:
                # print('slow down required, nextSpeed = ' + str(nextSpeed))
                # deceleration required
                decelerationDistance = getAccelerationDistance(speedDeltaNextSegment, currentSpeed)

                if decelerationDistance > currentDistance:
                    a = 0.5 * defaultMaxAcceleration
                    b = currentSpeed
                    c = -1.0 * currentDistance
                    d = (b ** 2) - (4 * a * c)

                    # print('a, b, c, d' + str(a) + ', ' + str(b) + ', ' + str(c) + ', ' + str(d))

                    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                    sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                    if type(sol1) == complex:
                        sol1 = sol1.real
                    if type(sol2) == complex:
                        sol2 = sol2.real
                    if sol1 <= 0:
                        sol = sol2
                    elif sol2 <= 0:
                        sol = sol1
                    elif sol1 < sol2:
                        sol = sol1
                    else:
                        sol = sol2
                    newSpeed = currentSpeed + (-1.0 * defaultMaxAcceleration * sol)
                    prevSpeed = newSpeed
                    # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(nextSpeed))
                    timeRequired = sol
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    restDistance = currentDistance - decelerationDistance
                    restTime = restDistance / currentSpeed
                    prevSpeed = nextSpeed
                    decelerationTime = abs(speedDeltaNextSegment / defaultMaxAcceleration)
                    timeRequired = decelerationTime + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        elif speedDeltaThisSegment > 0.0:
            # print('speed up required, prevSpeed = ' + str(prevSpeed) + ' and currentSpeed = ' + str(currentSpeed))
            # acceleration required to reach speed of current segment
            accelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)

            # given that acceleration happened previously, do we need to decelerate for next segment?
            speedDeltaNextSegment = nextSpeed - currentSpeed

            if speedDeltaNextSegment >= 0.0:
                # print('slow down not required, nextSpeed = ' + str(nextSpeed))
                # speed does not have to change, test if acceleration can be done within segment length
                if accelerationDistance > currentDistance:
                    # print('segment length too short for acceleration distance')
                    # segment not long enough, prevSpeed will not match currentSpeed
                    a = 0.5 * defaultMaxAcceleration
                    b = prevSpeed
                    c = -1.0 * currentDistance
                    d = (b ** 2) - (4 * a * c)
                    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                    sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                    if type(sol1) == complex:
                        sol1 = sol1.real
                    if type(sol2) == complex:
                        sol2 = sol2.real
                    if sol1 <= 0:
                        sol = sol2
                    elif sol2 <= 0:
                        sol = sol1
                    elif sol1 < sol2:
                        sol = sol1
                    else:
                        sol = sol2
                    newSpeed = prevSpeed + (defaultMaxAcceleration * sol)
                    prevSpeed = newSpeed
                    # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                    timeRequired = sol
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    # acceleration can be done within segment length, find restDistance / time
                    restDistance = currentDistance - abs(accelerationDistance)
                    restTime = restDistance / currentSpeed
                    prevSpeed = currentSpeed
                    accelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                    timeRequired = abs(accelerationTime) + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
            elif speedDeltaNextSegment < 0.0:
                # print('slow down required, nextSpeed = ' + str(nextSpeed))
                # we must decelerate before next segment (take into account the acceleration possible for current
                # segment aswell) can we accelerate, drive max speed, decelerate all within the given segment length?
                # do we have to accelerate less, in order to make the speed limit of the next segment?

                # acceleration required to reach speed of current segment
                accelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)

                # given that acceleration happened previously, do we need to decelerate for next segment?
                speedDeltaNextSegment = nextSpeed - currentSpeed

                decelerationDistance = getAccelerationDistance(speedDeltaNextSegment, currentSpeed)

                if accelerationDistance + decelerationDistance > currentDistance:
                    # print('segment length too short for acceleration distance + deceleration distance')
                    # can't make the acceleration deceleration combination within the segment length
                    # choose to only reach target speed for next segment -->

                    # need acceleration? or deceleration?
                    speedDeltaCrossSegment = nextSpeed - prevSpeed
                    if speedDeltaCrossSegment == 0.0:
                        # print('normal driven choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # drive at prevSpeed
                        timeRequired = currentDistance / prevSpeed
                        # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(nextSpeed))
                        # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        prevSpeed = nextSpeed
                    elif speedDeltaCrossSegment < 0.0:
                        # print('slow down choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # decelerate to nextspeed during the current segment
                        decelerationDistance = getAccelerationDistance(speedDeltaCrossSegment, prevSpeed)

                        if decelerationDistance > currentDistance:
                            a = 0.5 * defaultMaxAcceleration
                            b = prevSpeed
                            c = -1.0 * currentDistance
                            d = (b ** 2) - (4 * a * c)
                            sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                            sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                            if type(sol1) == complex:
                                sol1 = sol1.real
                            if type(sol2) == complex:
                                sol2 = sol2.real
                            if sol1 <= 0:
                                sol = sol2
                            elif sol2 <= 0:
                                sol = sol1
                            elif sol1 < sol2:
                                sol = sol1
                            else:
                                sol = sol2
                            newSpeed = prevSpeed + (-1.0 * defaultMaxAcceleration * sol)
                            prevSpeed = newSpeed
                            # print(
                            # 'leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                            timeRequired = sol
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        else:
                            restDistance = currentDistance - decelerationDistance
                            restTime = restDistance / currentSpeed
                            prevSpeed = nextSpeed
                            decelerationTime = abs(speedDeltaCrossSegment / defaultMaxAcceleration)
                            timeRequired = decelerationTime + restTime
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                    else:
                        # print('speed up choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # accelerate to nextspeed during the current segment
                        accelerationDistance = getAccelerationDistance(speedDeltaCrossSegment, prevSpeed)

                        if accelerationDistance > currentDistance:
                            # print('segment length too short for acceleration distance')
                            # segment not long enough, prevSpeed will not match currentSpeed
                            a = 0.5 * defaultMaxAcceleration
                            b = prevSpeed
                            c = -1.0 * currentDistance
                            d = (b ** 2) - (4 * a * c)
                            sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                            sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                            if type(sol1) == complex:
                                sol1 = sol1.real
                            if type(sol2) == complex:
                                sol2 = sol2.real
                            if sol1 <= 0:
                                sol = sol2
                            elif sol2 <= 0:
                                sol = sol1
                            elif sol1 < sol2:
                                sol = sol1
                            else:
                                sol = sol2
                            newSpeed = prevSpeed + (defaultMaxAcceleration * sol)
                            prevSpeed = newSpeed
                            # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(nextSpeed))
                            timeRequired = sol
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        else:
                            # acceleration can be done within segment length, find restDistance / time
                            restDistance = currentDistance - abs(accelerationDistance)
                            restTime = restDistance / nextSpeed
                            prevSpeed = nextSpeed
                            accelerationTime = abs(speedDeltaCrossSegment / defaultMaxAcceleration)
                            timeRequired = abs(accelerationTime) + restTime
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    # segment long enough to accelerate and decelerate again
                    accelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                    decelerationTime = abs(speedDeltaNextSegment / defaultMaxAcceleration)
                    restDistance = currentDistance - abs(accelerationDistance) - abs(decelerationDistance)
                    restTime = restDistance / currentSpeed
                    prevSpeed = nextSpeed
                    timeRequired = abs(accelerationTime) + abs(decelerationTime) + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        else:
            # deceleration required to reach current speed --> assuming this should never happen (given proper road
            # design)
            # print('deceleration required to reach the speed of the current segment prevSpeed = ' + str(prevSpeed) +
            # ' and currentSpeed = ' + str(currentSpeed))
            # print('speed delta: ' + str(speedDeltaThisSegment))
            decelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)
            # print('decelerationDistance = ' + str(decelerationDistance))

            if decelerationDistance > currentDistance:
                # print('deceleration distance too large for segment length')
                a = 0.5 * defaultMaxAcceleration
                b = prevSpeed
                c = -1.0 * currentDistance
                d = (b ** 2) - (4 * a * c)
                sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                if type(sol1) == complex:
                    sol1 = sol1.real
                if type(sol2) == complex:
                    sol2 = sol2.real
                if sol1 <= 0:
                    sol = sol2
                elif sol2 <= 0:
                    sol = sol1
                elif sol1 < sol2:
                    sol = sol1
                else:
                    sol = sol2
                newSpeed = prevSpeed + (-1.0 * defaultMaxAcceleration * sol)
                prevSpeed = newSpeed
                # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                timeRequired = sol
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))
            else:
                restDistance = currentDistance - decelerationDistance
                restTime = restDistance / currentSpeed
                prevSpeed = currentSpeed
                decelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                # print('decel time: ' + str(decelerationTime))
                # print('rest time: ' + str(restTime) + '  , for rest dist: ' + str(restDistance))
                timeRequired = decelerationTime + restTime
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        total_path_time_required_acceleration += timeRequired

    # print(' ')
    # print('-----------------------------------')
    # print('total time required default: ' + str(total_path_time_required))
    # print('total time required acceleration: ' + str(total_path_time_required_acceleration))
    # print('-----------------------------------')
    # print(' ')

    return total_path_time_required_acceleration


def node_list_to_path_time_required(G, node_list, firstCandidate, secondCandidate, crs, to_crs):
    node_list = node_list[:-1]
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    total_path_time_required = 0.0
    total_path_time_required_acceleration = 0.0
    distance_diff_total = 0.0

    distanceSpeedTuples = []

    for u, v in edge_nodes:
        # print(u, v)
        edge_data = G.get_edge_data(u, v)
        if edge_data is not None:
            if 'maxspeed' in edge_data[0]:
                if len(re.findall(r'\d+', str(edge_data[0]['maxspeed']))) > 0:
                    maxspeed = float(re.findall(r'\d+', str(edge_data[0]['maxspeed']))[0])
                    # print('FOUND IN OSM MAXSPEED: ' + str(maxspeed))
                    maxspeed = (maxspeed * 1.609344) / 3.6
                else:
                    maxspeed = (70 * 1.609344) / 3.6
            else:
                maxspeed = (70 * 1.609344) / 3.6

            # print(firstCandidate)
            # print(secondCandidate)
            if u == firstCandidate['from'] and v == firstCandidate['to']:
                length = edge_data[0]['length']
                # print('segment length: ' + str(length))
                p = transform_coordinates(firstCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                uNode = G.nodes[u]
                d = distance_between_points((uNode['y'], uNode['x']), projectionPoint)
                d = d * 1000
                distance_diff_total += d
                # print('distance from proj_point to u =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            elif u == secondCandidate['from'] and v == secondCandidate['to']:
                length = edge_data[0]['length']
                # print('segment length: ' + str(length))
                p = transform_coordinates(secondCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                vNode = G.nodes[v]
                d = distance_between_points((vNode['y'], vNode['x']), projectionPoint)
                d = d * 1000
                distance_diff_total += d
                # print('distance from proj_point to v =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            else:
                length = edge_data[0]['length']

            # print('MAXSPEED = ' + str(maxspeed))
            time_required_to_travel_the_road = length / maxspeed
            total_path_time_required += time_required_to_travel_the_road

            if maxspeed > 28.0:
                maxspeed = (70 * 1.609344) / 3.6

            distanceSpeedTuples.append((length, maxspeed))
        else:
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]

            if u == firstCandidate['from'] and v == firstCandidate['to']:
                length = distance_between_points(line[0], line[1])
                length = length * 1000
                # print('segment length: ' + str(length))
                p = transform_coordinates(firstCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                uNode = G.nodes[u]
                d = distance_between_points((uNode['y'], uNode['x']), projectionPoint)
                d = d * 1000
                distance_diff_total += d
                # print('distance from proj_point to u =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            elif u == secondCandidate['from'] and v == secondCandidate['to']:
                length = distance_between_points(line[0], line[1])
                length = length * 1000
                # print('segment length: ' + str(length))
                p = transform_coordinates(secondCandidate['proj_point'], crs, to_crs)
                projectionPoint = (p.y, p.x)
                vNode = G.nodes[v]
                d = distance_between_points((vNode['y'], vNode['x']), projectionPoint)
                d = d * 1000
                distance_diff_total += d
                # print('distance from proj_point to v =' + str(d))
                length = length - d
                # print('final length = ' + str(length))
            else:
                length = distance_between_points(line[0], line[1])
                length = length * 1000

            maxspeed = (70 * 1.609344) / 3.6
            time_required_to_travel_the_road = length / maxspeed
            total_path_time_required += time_required_to_travel_the_road
            # print('MAXSPEED = ' + str(maxspeed))
            distanceSpeedTuples.append((length, maxspeed))

    prevSpeed = 0.0
    # print('DISTANCE DIFF TOTAL = ' + str(distance_diff_total))
    # print('looking at a path with #' + str(len(distanceSpeedTuples)) + ' distance tuples')
    # print('length of node list: ' + str(len(node_list)))

    # default acceleration / deceleration in m/s^2
    defaultMaxAcceleration = 4.5
    for i in range(0, len(distanceSpeedTuples)):
        # print('i: ' + str(i))

        if i + 1 == len(distanceSpeedTuples):
            # print('driving the last segment:')
            timeRequired = distanceSpeedTuples[i][0] / distanceSpeedTuples[i][1]
            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(distanceSpeedTuples[i][0]))
            # print('default timeRequired = ' + str(distanceSpeedTuples[i][0] / distanceSpeedTuples[i][1]))
            total_path_time_required_acceleration += timeRequired
            break

        nextSpeed = distanceSpeedTuples[i + 1][1]
        currentSpeed = distanceSpeedTuples[i][1]
        if prevSpeed == 0.0:
            prevSpeed = currentSpeed

        currentDistance = distanceSpeedTuples[i][0]
        timeRequired = 0.0

        # speed diff between previous segment and current segment
        speedDeltaThisSegment = currentSpeed - prevSpeed
        if speedDeltaThisSegment == 0.0:
            # print('speed up not required, prevSpeed = ' + str(prevSpeed) + ' and currentSpeed = ' + str(currentSpeed))
            # current speed reached, figure out what to do to compensate for next segment if required
            speedDeltaNextSegment = nextSpeed - currentSpeed

            if speedDeltaNextSegment >= 0.0:
                # print('slow down not required, nextSpeed = ' + str(nextSpeed))
                # speed is fine, figure out time required to travel segment
                timeRequired = currentDistance / currentSpeed
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                prevSpeed = currentSpeed

            elif speedDeltaNextSegment < 0.0:
                # print('slow down required, nextSpeed = ' + str(nextSpeed))
                # deceleration required
                decelerationDistance = getAccelerationDistance(speedDeltaNextSegment, currentSpeed)

                if decelerationDistance > currentDistance:
                    a = 0.5 * defaultMaxAcceleration
                    b = currentSpeed
                    c = -1.0 * currentDistance
                    d = (b ** 2) - (4 * a * c)

                    # print('a, b, c, d' + str(a) + ', ' + str(b) + ', ' + str(c) + ', ' + str(d))

                    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                    sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                    if type(sol1) == complex:
                        sol1 = sol1.real
                    if type(sol2) == complex:
                        sol2 = sol2.real
                    if sol1 <= 0:
                        sol = sol2
                    elif sol2 <= 0:
                        sol = sol1
                    elif sol1 < sol2:
                        sol = sol1
                    else:
                        sol = sol2
                    newSpeed = currentSpeed + (-1.0 * defaultMaxAcceleration * sol)
                    prevSpeed = newSpeed
                    # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(nextSpeed))
                    timeRequired = sol
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    restDistance = currentDistance - decelerationDistance
                    restTime = restDistance / currentSpeed
                    prevSpeed = nextSpeed
                    decelerationTime = abs(speedDeltaNextSegment / defaultMaxAcceleration)
                    timeRequired = decelerationTime + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        elif speedDeltaThisSegment > 0.0:
            # print('speed up required, prevSpeed = ' + str(prevSpeed) + ' and currentSpeed = ' + str(currentSpeed))
            # acceleration required to reach speed of current segment
            accelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)

            # given that acceleration happened previously, do we need to decelerate for next segment?
            speedDeltaNextSegment = nextSpeed - currentSpeed

            if speedDeltaNextSegment >= 0.0:
                # print('slow down not required, nextSpeed = ' + str(nextSpeed))
                # speed does not have to change, test if acceleration can be done within segment length
                if accelerationDistance > currentDistance:
                    # print('segment length too short for acceleration distance')
                    # segment not long enough, prevSpeed will not match currentSpeed
                    a = 0.5 * defaultMaxAcceleration
                    b = prevSpeed
                    c = -1.0 * currentDistance
                    d = (b ** 2) - (4 * a * c)
                    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                    sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                    if type(sol1) == complex:
                        sol1 = sol1.real
                    if type(sol2) == complex:
                        sol2 = sol2.real
                    if sol1 <= 0:
                        sol = sol2
                    elif sol2 <= 0:
                        sol = sol1
                    elif sol1 < sol2:
                        sol = sol1
                    else:
                        sol = sol2
                    newSpeed = prevSpeed + (defaultMaxAcceleration * sol)
                    prevSpeed = newSpeed
                    # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                    timeRequired = sol
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    # acceleration can be done within segment length, find restDistance / time
                    restDistance = currentDistance - abs(accelerationDistance)
                    restTime = restDistance / currentSpeed
                    prevSpeed = currentSpeed
                    accelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                    timeRequired = abs(accelerationTime) + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))
            elif speedDeltaNextSegment < 0.0:
                # print('slow down required, nextSpeed = ' + str(nextSpeed))
                # we must decelerate before next segment (take into account the acceleration possible for current
                # segment aswell) can we accelerate, drive max speed, decelerate all within the given segment length?
                # do we have to accelerate less, in order to make the speed limit of the next segment?

                # acceleration required to reach speed of current segment
                accelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)

                # given that acceleration happened previously, do we need to decelerate for next segment?
                speedDeltaNextSegment = nextSpeed - currentSpeed

                decelerationDistance = getAccelerationDistance(speedDeltaNextSegment, currentSpeed)

                if accelerationDistance + decelerationDistance > currentDistance:
                    # print('segment length too short for acceleration distance + deceleration distance')
                    # can't make the acceleration deceleration combination within the segment length
                    # choose to only reach target speed for next segment -->

                    # need acceleration? or deceleration?
                    speedDeltaCrossSegment = nextSpeed - prevSpeed
                    if speedDeltaCrossSegment == 0.0:
                        # print('normal driven choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # drive at prevSpeed
                        timeRequired = currentDistance / prevSpeed
                        # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(nextSpeed))
                        # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        prevSpeed = nextSpeed
                    elif speedDeltaCrossSegment < 0.0:
                        # print('slow down choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # decelerate to nextspeed during the current segment
                        decelerationDistance = getAccelerationDistance(speedDeltaCrossSegment, prevSpeed)

                        if decelerationDistance > currentDistance:
                            a = 0.5 * defaultMaxAcceleration
                            b = prevSpeed
                            c = -1.0 * currentDistance
                            d = (b ** 2) - (4 * a * c)
                            sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                            sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                            if type(sol1) == complex:
                                sol1 = sol1.real
                            if type(sol2) == complex:
                                sol2 = sol2.real
                            if sol1 <= 0:
                                sol = sol2
                            elif sol2 <= 0:
                                sol = sol1
                            elif sol1 < sol2:
                                sol = sol1
                            else:
                                sol = sol2
                            newSpeed = prevSpeed + (-1.0 * defaultMaxAcceleration * sol)
                            prevSpeed = newSpeed
                            # print(
                            # 'leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                            timeRequired = sol
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        else:
                            restDistance = currentDistance - decelerationDistance
                            restTime = restDistance / currentSpeed
                            prevSpeed = nextSpeed
                            decelerationTime = abs(speedDeltaCrossSegment / defaultMaxAcceleration)
                            timeRequired = decelerationTime + restTime
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                    else:
                        # print('speed up choice, prevSpeed = ' + str(prevSpeed) + ' and nextSpeed = ' +
                        #       str(prevSpeed))
                        # accelerate to nextspeed during the current segment
                        accelerationDistance = getAccelerationDistance(speedDeltaCrossSegment, prevSpeed)

                        if accelerationDistance > currentDistance:
                            # print('segment length too short for acceleration distance')
                            # segment not long enough, prevSpeed will not match currentSpeed
                            a = 0.5 * defaultMaxAcceleration
                            b = prevSpeed
                            c = -1.0 * currentDistance
                            d = (b ** 2) - (4 * a * c)
                            sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                            sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                            if type(sol1) == complex:
                                sol1 = sol1.real
                            if type(sol2) == complex:
                                sol2 = sol2.real
                            if sol1 <= 0:
                                sol = sol2
                            elif sol2 <= 0:
                                sol = sol1
                            elif sol1 < sol2:
                                sol = sol1
                            else:
                                sol = sol2
                            newSpeed = prevSpeed + (defaultMaxAcceleration * sol)
                            prevSpeed = newSpeed
                            # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(nextSpeed))
                            timeRequired = sol
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                        else:
                            # acceleration can be done within segment length, find restDistance / time
                            restDistance = currentDistance - abs(accelerationDistance)
                            restTime = restDistance / nextSpeed
                            prevSpeed = nextSpeed
                            accelerationTime = abs(speedDeltaCrossSegment / defaultMaxAcceleration)
                            timeRequired = abs(accelerationTime) + restTime
                            # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                            # print('default timeRequired = ' + str(currentDistance / currentSpeed))
                else:
                    # segment long enough to accelerate and decelerate again
                    accelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                    decelerationTime = abs(speedDeltaNextSegment / defaultMaxAcceleration)
                    restDistance = currentDistance - abs(accelerationDistance) - abs(decelerationDistance)
                    restTime = restDistance / currentSpeed
                    prevSpeed = nextSpeed
                    timeRequired = abs(accelerationTime) + abs(decelerationTime) + restTime
                    # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                    # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        else:
            # deceleration required to reach current speed --> assuming this should never happen (given proper road
            # design)
            # print('deceleration required to reach the speed of the current segment prevSpeed = ' + str(prevSpeed) +
            # ' and currentSpeed = ' + str(currentSpeed))
            # print('speed delta: ' + str(speedDeltaThisSegment))
            decelerationDistance = getAccelerationDistance(speedDeltaThisSegment, prevSpeed)
            # print('decelerationDistance = ' + str(decelerationDistance))

            if decelerationDistance > currentDistance:
                # print('deceleration distance too large for segment length')
                a = 0.5 * defaultMaxAcceleration
                b = prevSpeed
                c = -1.0 * currentDistance
                d = (b ** 2) - (4 * a * c)
                sol1 = (-b - cmath.sqrt(d)) / (2 * a)
                sol2 = (-b + cmath.sqrt(d)) / (2 * a)
                if type(sol1) == complex:
                    sol1 = sol1.real
                if type(sol2) == complex:
                    sol2 = sol2.real
                if sol1 <= 0:
                    sol = sol2
                elif sol2 <= 0:
                    sol = sol1
                elif sol1 < sol2:
                    sol = sol1
                else:
                    sol = sol2
                newSpeed = prevSpeed + (-1.0 * defaultMaxAcceleration * sol)
                prevSpeed = newSpeed
                # print('leftover speed = ' + str(newSpeed) + ' , desired currentSpeed = ' + str(currentSpeed))
                timeRequired = sol
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))
            else:
                restDistance = currentDistance - decelerationDistance
                restTime = restDistance / currentSpeed
                prevSpeed = currentSpeed
                decelerationTime = abs(speedDeltaThisSegment / defaultMaxAcceleration)
                # print('decel time: ' + str(decelerationTime))
                # print('rest time: ' + str(restTime) + '  , for rest dist: ' + str(restDistance))
                timeRequired = decelerationTime + restTime
                # print('timeRequired = ' + str(timeRequired) + ' for dist: ' + str(currentDistance))
                # print('default timeRequired = ' + str(currentDistance / currentSpeed))

        total_path_time_required_acceleration += timeRequired

    # print(' ')
    # print('-----------------------------------')
    # print('total time required default: ' + str(total_path_time_required))
    # print('total time required acceleration: ' + str(total_path_time_required_acceleration))
    # print('-----------------------------------')
    # print(' ')

    return total_path_time_required_acceleration


def numberOfRoadTypeAlternations(G, node_list):
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    road_types = []
    for u, v in edge_nodes:
        edge_data = G.get_edge_data(u, v)
        if edge_data is not None:
            road_type = edge_data[0]['highway']
            road_types.append(road_type)
        else:
            road_types.append('')

    prev_value = 0
    alternations = 0
    is_first_time = True
    for road_type in road_types:
        if is_first_time:
            prev_value = road_type
            is_first_time = False
        else:
            if road_type != prev_value:
                alternations += 1
                prev_value = road_type
    return alternations


def pathIsPhysicallyRealizable(G, path, timeDelta, firstCandidate, secondCandidate, crs, to_crs):
    total_time_required = node_list_to_path_time_required(G, path, firstCandidate, secondCandidate, crs, to_crs)
    if total_time_required > timeDelta.total_seconds():
        return False
    return True


def getAllPathsForIndex(feasibleNetwork, index):
    outputSet = []
    for i in range(len(feasibleNetwork)):
        if feasibleNetwork[i][0] == index and feasibleNetwork[i] not in outputSet and len(feasibleNetwork[i]) > 0:
            outputSet.append(feasibleNetwork[i])
    return outputSet


def visualizeCandidatePathSegments(road_graph, feasibleNetwork):
    lats3 = []
    longs3 = []
    for i in range(len(feasibleNetwork)):
        firstSet = feasibleNetwork[i]
        paths = firstSet[5]

        lats2 = []
        longs2 = []
        for path in paths:
            lines = node_list_to_path(road_graph, path)
            long2 = []
            lat2 = []
            for j in range(len(lines)):
                z = list(lines[j])
                l1 = list(list(zip(*z))[0])
                l2 = list(list(zip(*z))[1])
                for k in range(len(l1)):
                    long2.append(l1[k])
                    lat2.append(l2[k])
            lats2.append(lat2)
            longs2.append(long2)
        lats3.append(lats2)
        longs3.append(longs2)


def visualizeCandidatePaths(road_graph, pathSet):
    lats2 = []
    longs2 = []
    for pathTuple in pathSet:
        lines = node_list_to_path(road_graph, pathTuple[1])
        long2 = []
        lat2 = []
        for j in range(len(lines)):
            z = list(lines[j])
            l1 = list(list(zip(*z))[0])
            l2 = list(list(zip(*z))[1])
            for k in range(len(l1)):
                long2.append(l1[k])
                lat2.append(l2[k])
        lats2.append(lat2)
        longs2.append(long2)

    plot_full_paths(lats2, longs2)


def pathToString(path):
    resultString = ''
    for node in path:
        resultString += str(node)
    return resultString


def doesPathAlreadyExist(feasibleNetwork, path):
    for resultTuple in feasibleNetwork:
        for p in resultTuple[5]:
            if pathToString(path) == pathToString(p):
                return True
    return False


def visualizeResultTuple(road_graph, resultTuple, i):
    lats2 = []
    longs2 = []
    for path in resultTuple[5]:
        lines = node_list_to_path(road_graph, path)
        long2 = []
        lat2 = []
        for j in range(len(lines)):
            z = list(lines[j])
            l1 = list(list(zip(*z))[0])
            l2 = list(list(zip(*z))[1])
            for k in range(len(l1)):
                long2.append(l1[k])
                lat2.append(l2[k])
        lats2.append(lat2)
        longs2.append(long2)

    plot_full_paths_numbered(lats2, longs2, i, resultTuple[5], road_graph)


def visualizeDebugPath(road_graph, path, i):
    lats2 = []
    longs2 = []
    lines = node_list_to_path(road_graph, path)
    long2 = []
    lat2 = []
    for j in range(len(lines)):
        z = list(lines[j])
        l1 = list(list(zip(*z))[0])
        l2 = list(list(zip(*z))[1])
        for k in range(len(l1)):
            long2.append(l1[k])
            lat2.append(l2[k])
    lats2.append(lat2)
    longs2.append(long2)

    plot_full_paths_numbered2(lats2, longs2, i, path, road_graph)


def createFeasibleNetwork(road_graph, trip, k, crs, to_crs, imageFolder):
    uniqueId = 0
    totalNetworks = []
    feasibleNetwork = []

    loopCounter = 0
    yesPathCounter = 0

    for i in range(len(trip) - 1):
        # print(' ')
        # print('I : ' + str(i))
        firstTripItem = trip.iloc[i]
        secondTripItem = trip.iloc[i + 1]

        timeDelta = secondTripItem['timestamp'] - firstTripItem['timestamp']
        noPathCounter = 0
        firstCandidateCounter = -1
        for firstCandidate in firstTripItem['candidates'].to_records(index=False):
            firstCandidateCounter += 1
            secondCandidateCounter = -1
            for secondCandidate in secondTripItem['candidates'].to_records(index=False):
                secondCandidateCounter += 1
                paths = k_shortest_paths(road_graph, firstCandidate['from'], secondCandidate['to'], k)
                allowedPaths = []
                if len(paths) == 0:
                    noPathCounter += 1
                    continue
                else:
                    # print('Generated k shortest paths from node ' + str(firstCandidate['from']) + ' to node ' +
                    #       str(secondCandidate['to']))
                    # print('With firstCandidate to: ' + str(firstCandidate['to']))
                    # print('With secondCandidate from: ' + str(secondCandidate['from']))
                    # print(paths)
                    # keep only the paths that contain from and to of both candidate points mapped segments
                    for path in paths:
                        if firstCandidate['from'] in path and firstCandidate['to'] in path and secondCandidate['from'] \
                                in path and secondCandidate['to'] in path:
                            allowedPaths.append(path)
                            yesPathCounter += 1

                    # print('Allowed k shortest paths from node ' + str(firstCandidate['from']) + ' to node ' +
                    #       str(secondCandidate['to']))
                    # print(allowedPaths)

                resultTuple = [loopCounter, loopCounter + 1, timeDelta, firstCandidate['from'], secondCandidate['to'],
                               allowedPaths, uniqueId, firstCandidate, secondCandidate, firstTripItem['timestamp'],
                               secondTripItem['timestamp']]

                visualizeGpsPointAndCandidatesCoupledPath(road_graph, trip, i, firstCandidate, secondCandidate,
                                                          firstCandidateCounter, secondCandidateCounter, allowedPaths,
                                                          crs, to_crs, imageFolder)
                uniqueId += 1
                feasibleNetwork.append(resultTuple)
        loopCounter += 1

        if noPathCounter == len(firstTripItem['candidates'].to_records(index=False)) \
                * len(secondTripItem['candidates'].to_records(index=False)):
            # print('For first candidates length: ' + str(len(firstTripItem['candidates'].to_records(index=False))))
            # print('And second candidates length: ' + str(len(secondTripItem['candidates'].to_records(index=False))))
            # print('The noPathCounter == ' + str(noPathCounter))
            # print(' ')
            # print(' ')
            totalNetworks.append(feasibleNetwork)
            feasibleNetwork = []
            loopCounter = 0

        # print('LOOPCOUNTER: ' + str(loopCounter))

    print('yespathcounter: ' + str(yesPathCounter))
    print('nr of points  : ' + str(len(trip)))

    totalNetworks.append(feasibleNetwork)

    return totalNetworks


# return a negative value (< 0) when the left item should be sorted before the right item
# return a positive value (> 0) when the left item should be sorted after the right item
# return 0 when both the left and the right item have the same weight and should be ordered "equally" without precedence
# (isRealizable, isBearingUniform, offrampAlternations, roadTypeAlternations, timeRequired, length, index)
def customComparator(item1, item2):
    timeRequired1 = item1[0]
    alternations1 = item1[1]

    timeRequired2 = item2[0]
    alternations2 = item2[1]

    weight1 = timeRequired1 * alternations1
    weight2 = timeRequired2 * alternations2

    if weight1 < weight2:
        return -1
    elif weight1 > weight2:
        return 1
    else:
        return 0


def customComparatorTuples(item1, item2):
    # print('comparator print')
    # print(item1)
    # print(item2)
    timeRequired1 = item1[2]
    alternations1 = item1[3]

    timeRequired2 = item2[2]
    alternations2 = item2[3]

    weight1 = timeRequired1 * alternations1
    weight2 = timeRequired2 * alternations2

    if weight1 < weight2:
        return -1
    elif weight1 > weight2:
        return 1
    else:
        return 0


def plotly_final_path(lat, long, imageFolder):
    """
    Given a list of latitudes and longitudes, origin
    and destination point, plots a path on a map

    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    import plotly.graph_objects as go
    import plotly
    import numpy as np

    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 13})
    fig.show()
    base_dir = Path(__file__).parent.resolve()
    uniqueId = str(lat_center).replace('.', '_')
    filePath = str(imageFolder) + '/plotly_' + uniqueId + '.html'
    fp = str(base_dir / filePath)
    plotly.offline.plot(fig, filename=fp, auto_open=False)


def visualizeFinalPaths(road_graph, path, imageFolder):
    lines = node_list_to_path(road_graph, path)
    long2 = []
    lat2 = []
    for i in range(len(lines)):
        z = list(lines[i])
        l1 = list(list(zip(*z))[0])
        l2 = list(list(zip(*z))[1])
        for j in range(len(l1)):
            long2.append(l1[j])
            lat2.append(l2[j])

    plotly_final_path(lat2, long2, imageFolder)


def isDuplicate(connectedPath, duplicateCheckingStructure):
    strPath = pathToString(connectedPath)
    return strPath in duplicateCheckingStructure
    # bucketId = strPath[:4]
    # if bucketId in duplicateCheckingStructure:
    #     for path in duplicateCheckingStructure[bucketId]:
    #         if path == strPath:
    #             return True
    # return False


def trimAllPossiblePathsGivenLength(road_graph, allPossiblePaths):
    # averageTimeRequired = 0.0
    # sumTimeRequired = 0.0
    # loopCounter = 0
    # for pathTuple in allPossiblePaths:
    #     # path = pathTuple[1]
    #     # timeRequired = node_list_to_path_time_required(road_graph, path)
    #     sumTimeRequired += pathTuple[2]
    #     loopCounter += 1
    # if loopCounter == 0:
    #     return allPossiblePaths
    # averageTimeRequired = sumTimeRequired / loopCounter
    # bestPossiblePaths = []
    # print('average time required: ' + str(averageTimeRequired))
    # for pathTuple in allPossiblePaths:
    #     # path = pathTuple[1]
    #     # pathTuple -> (firstCandidate, paths, time_required)
    #     # timeRequired = node_list_to_path_time_required(road_graph, path)
    #     if pathTuple[2] <= averageTimeRequired:
    #         bestPossiblePaths.append(pathTuple)
    # return bestPossiblePaths
    # print(allPossiblePaths)
    allPossiblePaths.sort(key=cmp_to_key(customComparatorTuples))
    # bestPossiblePaths = sorted(allPossiblePaths, key=cmp_to_key(customComparatorTuples))
    # print(bestPossiblePaths)
    return allPossiblePaths[:min(250, len(allPossiblePaths))]


def pathIsContained(path1, path2):
    for path in path2:
        if path not in path1:
            return False
    return True


def insertIntoAllPossiblePaths(allPossiblePaths, tup):
    wasInserted = False
    for i in range(0, len(allPossiblePaths)):
        if allPossiblePaths[i][2] >= tup[2]:
            allPossiblePaths.insert(i, tup)
            wasInserted = True
            break
    if not wasInserted:
        allPossiblePaths.append(tup)
    return allPossiblePaths[:min(250, len(allPossiblePaths))]


def solveForGivenNetwork(trip, road_graph, feasibleNetwork, crs, to_crs):
    duplicate_counter = 0
    subpath_counter = 0
    appended_counter = 0

    # print('Appending first candidate set to the all possible path list..')
    # append all 0 -> 1 candidates paths to the list
    allPossiblePaths = []
    firstCandidateSet = getAllPathsForIndex(feasibleNetwork, 0)
    for firstCandidate in firstCandidateSet:
        firstCandidatePaths = firstCandidate[5]
        for firstCandidatePath in firstCandidatePaths:
            # resultTuple = [loopCounter, loopCounter + 1, timeDelta, firstCandidate['from'], secondCandidate['to'],
            #                allowedPaths, uniqueId, firstCandidate, secondCandidate, firstTripItem['timestamp'],
            #                secondTripItem['timestamp']]
            allPossiblePaths.append((firstCandidate[7], firstCandidatePath, 0.0, 1))

    # print('Given the init list of possible paths of length = ' + str(len(allPossiblePaths)))

    isFirstTime = True
    duplicateCheckingStructure = {}
    duplicateList = []
    duplicateCheckSet = set()
    initTimeStamp = trip.iloc[0]['timestamp']
    # print(initTimeStamp)

    # incrementally increase the path length by looking at the next candidate set
    for i in range(1, len(trip) - 1):

        if not isFirstTime:
            # print('I (' + str(i) + ') - allPossiblePaths length before trim: ' + str(len(allPossiblePaths)))
            allPossiblePaths = trimAllPossiblePathsGivenLength(road_graph, allPossiblePaths)
            # print('I (' + str(i) + ') - allPossiblePaths length after trim: ' + str(len(allPossiblePaths)))
        else:
            isFirstTime = False

        secondTimeStamp = trip.iloc[i + 1]['timestamp']
        timeDeltaUpToNow = secondTimeStamp - initTimeStamp
        # print(secondTimeStamp)
        # print(initTimeStamp)
        # print(timeDeltaUpToNow)
        secondCandidateSet = getAllPathsForIndex(feasibleNetwork, i)
        # print('length of allpossible paths before deepcopy: ' + str(len(allPossiblePaths)))
        allPossiblePathsImmutable = copy.deepcopy(allPossiblePaths)
        allPossiblePaths = []
        for firstCandidatePathTuple in allPossiblePathsImmutable:
            # print(firstCandidatePathTuple)
            firstCandidatePath = firstCandidatePathTuple[1]
            # print(firstCandidatePath)
            firstCandidate = firstCandidatePathTuple[0]

            # print(firstCandidate)
            for secondCandidate in secondCandidateSet:
                secondCandidatePaths = secondCandidate[5]
                for secondCandidatePath in secondCandidatePaths:
                    wasSubPath = False

                    connectedPath = []
                    if len(firstCandidatePath) > 0 and len(secondCandidatePath) > 0 \
                            and firstCandidatePath[-1] == secondCandidatePath[0]:
                        # print('connecting -->')
                        # print(firstCandidatePath)
                        # print(secondCandidatePath)
                        # print('to -->')
                        connectedPath = firstCandidatePath[:-1] + secondCandidatePath
                        # print(connectedPath)
                    if len(firstCandidatePath) > 0 and len(secondCandidatePath) > 0 \
                            and firstCandidatePath[-2] == secondCandidatePath[0]:
                        # print('connecting -->')
                        # print(firstCandidatePath)
                        # print(secondCandidatePath)
                        # print('to -->')
                        connectedPath = firstCandidatePath[:-2] + secondCandidatePath
                        # print(connectedPath)
                    if len(firstCandidatePath) > 0 and len(secondCandidatePath) > 0 \
                            and pathIsContained(firstCandidatePath, secondCandidatePath):
                        # print('SUBPATH connecting -->')
                        # print(firstCandidatePath)
                        # print(secondCandidatePath)
                        # print('to -->')
                        wasSubPath = True
                        connectedPath = firstCandidatePath[:-2] + secondCandidatePath
                        # print(connectedPath)

                    if len(connectedPath) > 0:
                        time_required_acceleration = node_list_to_path_time_required(road_graph, connectedPath,
                                                                                     firstCandidate, secondCandidate[7],
                                                                                     crs, to_crs)

                        # pathIsPhysicallyRealizable(road_graph, connectedPath, timeDeltaUpToNow, firstCandidate,
                        #                            secondCandidate[7], crs, to_crs)

                        # print(time_required_acceleration)
                        if time_required_acceleration <= timeDeltaUpToNow.total_seconds():
                            # tempTest = pathIsPhysicallyRealizable(road_graph, connectedPath, timeDeltaUpToNow,
                            #                                       firstCandidate, secondCandidate[7], crs, to_crs)
                            roadTypeAlternations = numberOfRoadTypeAlternations(road_graph, connectedPath)

                            if wasSubPath:
                                # allPossiblePaths = insertIntoAllPossiblePaths(allPossiblePaths,
                                #                                               (firstCandidate, connectedPath,
                                #                                                time_required_acceleration,
                                #                                                roadTypeAlternations))
                                allPossiblePaths.append((firstCandidate, connectedPath, time_required_acceleration,
                                                         roadTypeAlternations))
                                subpath_counter += 1
                                appended_counter += 1
                            else:
                                prevSetSize = len(duplicateCheckSet)
                                strPath = pathToString(connectedPath)
                                duplicateCheckSet.add(strPath)
                                if (len(duplicateCheckSet)) > prevSetSize:
                                    # if not isDuplicate(connectedPath, duplicateCheckingStructure):
                                    # if not isDuplicate(connectedPath, duplicateList):
                                    # print((firstCandidate, connectedPath, time_required_acceleration))
                                    allPossiblePaths.append((firstCandidate, connectedPath, time_required_acceleration,
                                                             roadTypeAlternations))

                                    appended_counter += 1
                                    # allPossiblePaths = insertIntoAllPossiblePaths(allPossiblePaths,
                                    #                                               (firstCandidate, connectedPath,
                                    #                                                time_required_acceleration,
                                    #                                                roadTypeAlternations))
                                    # strPath = pathToString(connectedPath)
                                    # duplicateList.append(strPath)
                                    # bucketId = strPath[:4]
                                    # if bucketId in duplicateCheckingStructure:
                                    #     duplicateCheckingStructure[bucketId].append(strPath)
                                    # else:
                                    #     duplicateCheckingStructure[bucketId] = [strPath]
                                else:
                                    duplicate_counter += 1
                                # print('duplicate..')
                    #     else:
                    #         print('connected path not physically realizable.. (' + str(i) + ')')
                    # else:
                    #     print('empty connected path..')

    # print('Number of candidates in feasibleNetwork: ' + str(len(feasibleNetwork)))

    print('subpath counter == ' + str(subpath_counter))
    print('duplicate counter == ' + str(duplicate_counter))
    print('appended counter == ' + str(appended_counter))

    totalTimeDelta = trip.iloc[len(trip) - 1]['timestamp'] - trip.iloc[0]['timestamp']
    realizablePaths = allPossiblePaths

    visualizeCandidatePaths(road_graph, realizablePaths)

    finalCandidates = []
    index = 0
    for pathTuple in realizablePaths:
        path = pathTuple[1]
        timeRequired = pathTuple[2]
        # timeRequired = node_list_to_path_time_required(road_graph, path)
        roadTypeAlternations = numberOfRoadTypeAlternations(road_graph, path)
        comparePackage = (timeRequired, roadTypeAlternations, index, path)
        finalCandidates.append(comparePackage)

        # print('--------- path ' + str(index) + ' ---------')
        # print('timeRequired: ' + str(timeRequired) + ' seconds')
        # print('roadTypeAlternations: ' + str(roadTypeAlternations))
        index += 1

    finalCandidates = sorted(finalCandidates, key=cmp_to_key(customComparator))

    # print('Number of realizable paths: ' + str(len(realizablePaths)))
    # print(' ')
    if len(finalCandidates) > 0:
        # print('Best candidate: ')
        # print(finalCandidates[0])
        return finalCandidates[0]
    else:
        return 0


def visualize_by_folium(road_graph, trip, crs, to_crs):
    (minx, miny, maxx, maxy) = trip_bbox(trip)
    lon = (maxx + minx) / 2.0
    lat = (maxy + miny) / 2.0
    foliumMap = folium.Map(location=[lat, lon], zoom_start=15)

    for row in trip.to_records(index=False):
        folium.Marker((row['lat'], row['lon']), icon=folium.Icon(color='red')).add_to(foliumMap)
        tripCandidates = row['candidates']
        for candidate in tripCandidates.to_records(index=False):
            p = transform_coordinates(candidate['proj_point'], crs, to_crs)
            projectionPoint = (p.y, p.x)
            tooltip = 'Click me!'
            # folium.Marker((fromNode['y'], fromNode['x']), icon=folium.Icon(color='green'),
            #               popup='<i>From Node: ' + str(row['timestamp']) + '</i>',
            #               tooltip=tooltip).add_to(foliumMap)
            # folium.Marker((toNode['y'], fromNode['x']), icon=folium.Icon(color='beige'),
            #               popup='<i>To Node: ' + str(row['timestamp']) + '</i>',
            #               tooltip=tooltip).add_to(foliumMap)
            folium.Marker(projectionPoint, icon=folium.Icon(color='black'),
                          popup='<i>Candidate: ' + str(row['timestamp']) + '</i>',
                          tooltip=tooltip).add_to(foliumMap)
            polyline = [(row['lat'], row['lon']), projectionPoint]
            folium.PolyLine(polyline, color='red', weight=2.5, opacity=1).add_to(foliumMap)

            # folium.PolyLine([(fromNode['y'], fromNode['x']), projectionPoint], color='blue', weight=2.5, opacity=1)\
            #     .add_to(foliumMap)
            # folium.PolyLine([(toNode['y'], fromNode['x']), (fromNode['y'], fromNode['x'])], color='orange', weight=2.5, opacity=1)\
            #     .add_to(foliumMap)

    base_dir = Path(__file__).parent.resolve()
    foliumMap.save(str(base_dir / 'foliumMap.html'))

    return foliumMap


def visualizeGpsPointsAndCandidatesIndividual(road_graph, trip, crs, to_crs, imageFolder):
    (minx, miny, maxx, maxy) = trip_bbox(trip)
    lon = (maxx + minx) / 2.0
    lat = (maxy + miny) / 2.0
    tooltip = 'Click me!'
    tripItemCounter = 0
    for tripItem in trip.to_records(index=False):
        candidateCounter = 0
        tripCandidates = tripItem['candidates']
        for candidate in tripCandidates.to_records(index=False):
            foliumMap = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker((tripItem['lat'], tripItem['lon']), icon=folium.Icon(color='red')).add_to(foliumMap)
            p = transform_coordinates(candidate['proj_point'], crs, to_crs)
            projectionPoint = (p.y, p.x)
            fromPoint = road_graph.nodes[candidate['from']]
            toPoint = road_graph.nodes[candidate['to']]
            folium.Marker(projectionPoint, icon=folium.Icon(color='black'),
                          popup='<i>Candidate (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            folium.Marker((fromPoint['y'], fromPoint['x']), icon=folium.Icon(color='green'),
                          popup='<i>From (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            folium.Marker((toPoint['y'], toPoint['x']), icon=folium.Icon(color='blue'),
                          popup='<i>To (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            base_dir = Path(__file__).parent.resolve()
            filepath = str(imageFolder) + 'individualCandidates/tripItem-' + str(tripItemCounter) + \
                       '_candidate-' + str(candidateCounter) + '.html'
            foliumMap.save(str(base_dir / filepath))
            candidateCounter += 1
        tripItemCounter += 1


def visualizeGpsPointsAndCandidatesGrouped(road_graph, trip, crs, to_crs, imageFolder):
    (minx, miny, maxx, maxy) = trip_bbox(trip)
    lon = (maxx + minx) / 2.0
    lat = (maxy + miny) / 2.0
    tooltip = 'Click me!'
    tripItemCounter = 0
    for tripItem in trip.to_records(index=False):
        foliumMap = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker((tripItem['lat'], tripItem['lon']), icon=folium.Icon(color='red')).add_to(foliumMap)
        candidateCounter = 0
        tripCandidates = tripItem['candidates']
        for candidate in tripCandidates.to_records(index=False):
            p = transform_coordinates(candidate['proj_point'], crs, to_crs)
            projectionPoint = (p.y, p.x)
            fromPoint = road_graph.nodes[candidate['from']]
            toPoint = road_graph.nodes[candidate['to']]
            folium.Marker(projectionPoint, icon=folium.Icon(color='black'),
                          popup='<i>Candidate (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            folium.Marker((fromPoint['y'], fromPoint['x']), icon=folium.Icon(color='green'),
                          popup='<i>From (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            folium.Marker((toPoint['y'], toPoint['x']), icon=folium.Icon(color='blue'),
                          popup='<i>To (' + str(tripItemCounter) + ', ' + str(candidateCounter) + '): ' +
                                str(tripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
            candidateCounter += 1
        base_dir = Path(__file__).parent.resolve()
        filepath = str(imageFolder) + 'groupedByTripItem/tripItem-' + str(tripItemCounter) + \
                   '_candidate-' + str(candidateCounter) + '.html'
        foliumMap.save(str(base_dir / filepath))
        tripItemCounter += 1


def visualizeGpsPointAndCandidatesCoupledPath(road_graph, trip, i, firstCandidate, secondCandidate,
                                              firstCandidateCounter, secondCandidateCounter, paths, crs, to_crs,
                                              imageFolder):
    (minx, miny, maxx, maxy) = trip_bbox(trip)
    lon = (maxx + minx) / 2.0
    lat = (maxy + miny) / 2.0
    tooltip = 'Click me!'
    tripItemCounter = 0
    firstTripItem = trip.iloc[i]
    secondTripItem = trip.iloc[i + 1]
    foliumMap = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker((firstTripItem['lat'], firstTripItem['lon']), icon=folium.Icon(color='red')).add_to(foliumMap)
    folium.Marker((secondTripItem['lat'], secondTripItem['lon']), icon=folium.Icon(color='red')).add_to(foliumMap)

    p = transform_coordinates(firstCandidate['proj_point'], crs, to_crs)
    projectionPoint = (p.y, p.x)
    fromPoint = road_graph.nodes[firstCandidate['from']]
    toPoint = road_graph.nodes[firstCandidate['to']]
    folium.Marker(projectionPoint, icon=folium.Icon(color='black'),
                  popup='<i>Candidate (' + str(tripItemCounter) + ', ' + str(firstCandidateCounter) + '): ' +
                        str(firstTripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
    folium.Marker((fromPoint['y'], fromPoint['x']), icon=folium.Icon(color='green'),
                  popup='<i>From ' + str(fromPoint['osmid']) + ' (' + str(tripItemCounter) + ', ' +
                        str(firstCandidateCounter) + '): ' + str(firstTripItem['timestamp']) + '</i>',
                  tooltip=tooltip).add_to(foliumMap)
    folium.Marker((toPoint['y'], toPoint['x']), icon=folium.Icon(color='blue'),
                  popup='<i>To ' + str(toPoint['osmid']) + ' (' + str(tripItemCounter) + ', ' +
                        str(firstCandidateCounter) + '): ' + str(firstTripItem['timestamp']) + '</i>',
                  tooltip=tooltip).add_to(foliumMap)

    p = transform_coordinates(secondCandidate['proj_point'], crs, to_crs)
    projectionPoint = (p.y, p.x)
    fromPoint = road_graph.nodes[secondCandidate['from']]
    toPoint = road_graph.nodes[secondCandidate['to']]
    folium.Marker(projectionPoint, icon=folium.Icon(color='black'),
                  popup='<i>Candidate (' + str(tripItemCounter) + ', ' + str(secondCandidateCounter) + '): ' +
                        str(firstTripItem['timestamp']) + '</i>', tooltip=tooltip).add_to(foliumMap)
    folium.Marker((fromPoint['y'], fromPoint['x']), icon=folium.Icon(color='green'),
                  popup='<i>From ' + str(fromPoint['osmid']) + ' (' + str(tripItemCounter) + ', ' +
                        str(secondCandidateCounter) + '): ' + str(firstTripItem['timestamp']) + '</i>',
                  tooltip=tooltip).add_to(foliumMap)
    folium.Marker((toPoint['y'], toPoint['x']), icon=folium.Icon(color='blue'),
                  popup='<i>To ' + str(toPoint['osmid']) + ' (' + str(tripItemCounter) + ', ' +
                        str(secondCandidateCounter) + '): ' + str(firstTripItem['timestamp']) + '</i>',
                  tooltip=tooltip).add_to(foliumMap)

    for path in paths:
        lines = node_list_to_path(road_graph, path)
        longs = []
        lats = []
        for j in range(len(lines)):
            z = list(lines[j])
            l1 = list(list(zip(*z))[0])
            l2 = list(list(zip(*z))[1])
            for k in range(len(l1)):
                longs.append(l1[k])
                lats.append(l2[k])
        polyline = []
        for j in range(len(lats)):
            tup = (lats[j], longs[j])
            polyline.append(tup)
        if len(polyline) > 1:
            folium.PolyLine(polyline, color='blue', weight=2.5, opacity=1).add_to(foliumMap)

    base_dir = Path(__file__).parent.resolve()
    filepath = str(imageFolder) + 'coupledCandidatePaths/trip_' + str(i) + '_to_' + str(i + 1) + '_candidates' + \
               str(firstCandidateCounter) + '_and_' + str(secondCandidateCounter) + '.html'
    foliumMap.save(str(base_dir / filepath))


def clearDirectory(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def readNetwork(boundingBoxPolygonFile='E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
                tripFileName='E:/Thesis TUe/BerryLib/test_trajectories/traj11_div3.csv',
                imageFolder='candidateVisualizations/traj11_div1/'):
    start_datetime = datetime.now()
    start_time = start_datetime.strftime("%H:%M:%S")
    print("Start DateTime = ", start_datetime)

    # administration part
    base_dir = Path(__file__).parent.resolve()
    clearFolder = base_dir / imageFolder
    clearDirectory(str(clearFolder / 'individualCandidates/*'))
    clearDirectory(str(clearFolder / 'groupedByTripItem/*'))
    clearDirectory(str(clearFolder / 'coupledCandidatePaths/*'))

    after_administration_datetime = datetime.now()
    after_administration_time = after_administration_datetime.strftime("%H:%M:%S")
    print("After administration DateTime = ", after_administration_datetime)

    # step 1 --> read csv
    trip = read_trip(tripFileName)

    after_step1_datetime = datetime.now()
    after_step1_time = after_step1_datetime.strftime("%H:%M:%S")
    print("After Step1 DateTime = ", after_step1_datetime)

    # step 2 --> get road graph
    df = pd.read_csv(boundingBoxPolygonFile)
    boundingBoxPolygon = Polygon(list(zip(df.lon, df.lat)))
    road_graph = ox.graph_from_polygon(boundingBoxPolygon, network_type='drive', simplify=True, retain_all=True,
                                       truncate_by_edge=True)
    road_graph_utm = ox.project_graph(road_graph)
    gpd_edges_utm = road_graph_to_edge_gpd(road_graph_utm)
    max_speeds = get_max_speeds(gpd_edges_utm)
    gpd_edges_utm['max speed'] = max_speeds
    edge_idx = build_rtree_index_edges(gpd_edges_utm)
    crs = road_graph.graph['crs']
    to_crs = road_graph_utm.graph['crs']
    trip['geometry_utm'] = trip.apply(lambda row: transform_coordinates(row['geometry'], crs, to_crs), axis=1)

    after_step2_datetime = datetime.now()
    after_step2_time = after_step2_datetime.strftime("%H:%M:%S")
    print("After Step2 DateTime = ", after_step2_datetime)

    # step 3 --> find candidates
    k = 10
    find_candidates(trip, edge_idx, k)
    crs_utm = road_graph_utm.graph['crs']
    crs_wgs84 = road_graph.graph['crs']

    visualizeGpsPointsAndCandidatesIndividual(road_graph, trip, crs_utm, crs_wgs84, imageFolder)
    visualizeGpsPointsAndCandidatesGrouped(road_graph, trip, crs_utm, crs_wgs84, imageFolder)

    after_step3_datetime = datetime.now()
    after_step3_time = after_step3_datetime.strftime("%H:%M:%S")
    print("After Step3 DateTime = ", after_step3_datetime)

    # step 4 --> connect consecutive candidates
    totalNetworks = createFeasibleNetwork(road_graph, trip, 5, crs_utm, crs_wgs84, imageFolder)

    after_step4_datetime = datetime.now()
    after_step4_time = after_step4_datetime.strftime("%H:%M:%S")
    print("After Step4 DateTime = ", after_step4_datetime)

    # step 5 --> incrementally form (sub-)trajectories
    # print('created totalNetworks, testing feasibleNetworks 1 by 1...')
    finalCandidates = []
    for feasibleNetwork in totalNetworks:
        # print('feasibleNetwork has length: ' + str(len(feasibleNetwork)))
        finalCandidate = solveForGivenNetwork(trip, road_graph, feasibleNetwork, crs_utm, crs_wgs84)
        # print('finalCandidate value')
        # print(finalCandidate)
        if finalCandidate == 0:
            continue
        else:
            finalCandidates.append(finalCandidate)

    after_step5_datetime = datetime.now()
    after_step5_time = after_step5_datetime.strftime("%H:%M:%S")
    print("After Step5 DateTime = ", after_step5_datetime)

    # step 6 --> find best trajectory
    for finalCandidate in finalCandidates:
        visualizeFinalPaths(road_graph, finalCandidate[3], imageFolder)

    after_step6_datetime = datetime.now()
    after_step6_time = after_step6_datetime.strftime("%H:%M:%S")
    print("After Step6 DateTime = ", after_step6_datetime)


batchTodoItems = [
    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj11_div3_exp.csv',
    #  'candidateVisualizations/traj11_div3_exp/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj11_div3_exp2.csv',
    #  'candidateVisualizations/traj11_div3_exp2/')

    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj11_div1.csv',
    #  'candidateVisualizations/traj11_div1/'),
    #
    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj11_div2.csv',
    #  'candidateVisualizations/traj11_div2/'),
    #
    ('E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
     'E:/Thesis TUe/BerryLib/test_trajectories/traj11_div3.csv',
     'candidateVisualizations/traj11_div3/')
    # ,
    #
    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj10_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj10_div1.csv',
    #  'candidateVisualizations/traj10_div1/'),
    #
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj7_part1.csv',
    #  'candidateVisualizations/traj7_part1/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj3_part1.csv',
    #  'candidateVisualizations/traj3_part1/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj5_part1.csv',
    #  'candidateVisualizations/traj5_part1/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj6_part3.csv',
    #  'candidateVisualizations/traj6_part3/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj4_part3.csv',
    #  'candidateVisualizations/traj4_part3/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_1.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj0_part3.csv',
    #  'candidateVisualizations/traj0_part3/'),
    #
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj7_part3.csv',
    #  'candidateVisualizations/traj7_part3/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj3_part3.csv',
    #  'candidateVisualizations/traj3_part3/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj5_part3.csv',
    #  'candidateVisualizations/traj5_part3/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj6_part1.csv',
    #  'candidateVisualizations/traj6_part1/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj4_part1.csv',
    #  'candidateVisualizations/traj4_part1/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_2.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj0_part1.csv',
    #  'candidateVisualizations/traj0_part1/'),
    #
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj7_part2.csv',
    #  'candidateVisualizations/traj7_part2/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj3_part2.csv',
    #  'candidateVisualizations/traj3_part2/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj5_part2.csv',
    #  'candidateVisualizations/traj5_part2/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj6_part2.csv',
    #  'candidateVisualizations/traj6_part2/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj4_part2.csv',
    #  'candidateVisualizations/traj4_part2/'),
    # ('E:/Thesis TUe/BerryLib/test_trajectories/multi_bb_3.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj0_part2.csv',
    #  'candidateVisualizations/traj0_part2/'
    # ('E:/Thesis TUe/BerryLib/test_trajectories/traj10_test_bb.csv',
    #  'E:/Thesis TUe/BerryLib/test_trajectories/traj10_test.csv',
    #  'candidateVisualizations/traj10_test/')
]

for tup in batchTodoItems:
    print(tup)
    readNetwork(boundingBoxPolygonFile=tup[0], tripFileName=tup[1], imageFolder=tup[2])

# readNetwork(boundingBoxPolygonFile='E:/Thesis TUe/BerryLib/test_trajectories/traj11_bb.csv',
#             tripFileName='E:/Thesis TUe/BerryLib/test_trajectories/traj11_div1.csv',
#             imageFolder='candidateVisualizations/traj11_div1/')
