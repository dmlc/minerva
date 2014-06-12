#ifndef MINERVA_UTIL_GRAPH_H
#define MINERVA_UTIL_GRAPH_H
#pragma once

#include <cassert>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include <boost/tuple/tuple.hpp>

/* definition of basic graph properties */
enum vertex_properties_t { vertex_properties };
enum edge_properties_t { edge_properties };
namespace boost 
{
	BOOST_INSTALL_PROPERTY(vertex, properties);
	BOOST_INSTALL_PROPERTY(edge, properties);
}

namespace minerva
{
namespace utils
{
	using namespace boost;
	/* the graph base class template */
	template <typename VID, typename VProp, typename EID, typename EProp>
	class Graph
	{
	public:
		/* an adjacency_list like we need it */
		typedef adjacency_list<
			setS, // disallow parallel edges
			vecS, // vertex container
			bidirectionalS, // undirected graph
			// vertex properties
			property<vertex_name_t, VID,
			property<vertex_properties_t, VProp> > ,
			// edge properties
			property<edge_name_t, EID, 
			property<edge_properties_t, EProp> >
		> GraphContainer;


		/* property map typedefs */
		typedef typename property_map<GraphContainer, vertex_name_t>::type VIDMap;
		typedef typename property_map<GraphContainer, vertex_properties_t>::type VPropMap;
		typedef typename property_map<GraphContainer, edge_name_t>::type EIDMap;
		typedef typename property_map<GraphContainer, edge_properties_t>::type EPropMap;
		/* a bunch of graph-specific typedefs */
		typedef graph_traits<GraphContainer> GraphTraits;
		typedef typename GraphTraits::vertex_descriptor Vertex;
		typedef typename GraphTraits::edge_descriptor Edge;
		typedef std::pair<Edge, Edge> EdgePair;

		typedef typename GraphTraits::vertex_iterator VertexIter;
		typedef typename GraphTraits::edge_iterator EdgeIter;
		typedef typename GraphTraits::adjacency_iterator AdjacencyIter;
		typedef typename inv_adjacency_iterator_generator<GraphContainer>::type  InvAdjacencyIter;
		typedef typename GraphTraits::out_edge_iterator OutEdgeIter;
		typedef typename GraphTraits::in_edge_iterator InEdgeIter;

		typedef typename GraphTraits::vertices_size_type VerticesSize;
		typedef typename GraphTraits::edges_size_type EdgesSize;
		typedef typename GraphTraits::degree_size_type DegreeSize;

		typedef std::pair<AdjacencyIter, AdjacencyIter> AdjacencyVertexRange;
		typedef std::pair<InvAdjacencyIter, InvAdjacencyIter> InvAdjacencyVertexRange;
		typedef std::pair<OutEdgeIter, OutEdgeIter> OutEdgeRange;
		typedef std::pair<InEdgeIter, InEdgeIter> InEdgeRange;
		typedef std::pair<VertexIter, VertexIter> VertexRange;
		typedef std::pair<EdgeIter, EdgeIter> EdgeRange;

	public:
		/* constructors etc. */
		Graph()
		{}

		Graph(const Graph& g):
			graph(g.graph)
		{}

		virtual ~Graph()
		{}


		/* structure modification methods */
		void Clear()
		{
			graph.clear();
		}

		Vertex AddVertex(const VID& id, const VProp& prop)
		{
			Vertex v = add_vertex(graph);
			Id(v) = id;
			Properties(v) = prop;
			return v;
		}

		void RemoveVertex(const Vertex& v)
		{
			clear_vertex(v, graph);
			remove_vertex(v, graph);
		}

		Edge AddEdge(const Vertex& v1, const Vertex& v2, const EID& id, const EProp& prop)
		{
			Edge e = add_edge(v1, v2, graph).first;
			Id(e) = id;
			Properties(e) = prop;
			return e;
		}

		/* vertex ids*/
		VID& Id(const Vertex& v) { return get(vertex_name, graph)[v]; }
		const VID& Id(const Vertex& v) const { return get(vertex_name, graph)[v]; }
		/* vertex property access */
		VProp& Properties(const Vertex& v) { return get(vertex_properties, graph)[v]; }
		const VProp& Properties(const Vertex& v) const { return get(vertex_properties, graph)[v]; }

		/* edge ids*/
		EID& Id(const Edge& e) { return get(edge_name, graph)[e]; }
		const EID& Id(const Edge& e) const { return get(edge_name, graph)[e]; }
		/* edge property access */
		EProp& Properties(const Edge& e) { return get(edge_properties, graph)[e]; }
		const EProp& Properties(const Edge& e) const { return get(edge_properties, graph)[e]; }
		EProp& Properties(const Vertex& src, const Vertex& tgt)
		{
			Edge e;
			bool exist;
			tie(e, exist) = edge(src, tgt, graph);
			assert(exist); // Edge must exist
			return get(edge_properties, graph)[e];
		}
		const EProp& Properties(const Vertex& src, const Vertex& tgt) const
		{
			Edge e;
			bool exist;
			tie(e, exist) = edge(src, tgt, graph);
			assert(exist); // Edge must exist
			return get(edge_properties, graph)[e];
		}

		/* selectors and properties */
		const GraphContainer& getGraph() const { return graph; }
		VertexRange Vertices() const { return vertices(graph); }
		EdgeRange Edges() const { return edges(graph); }
		InvAdjacencyVertexRange InvAdjacentVertices(const Vertex& v) const
		{
			return inv_adjacent_vertices(v, graph);
		}
		AdjacencyVertexRange AdjacentVertices(const Vertex& v) const
		{
			return adjacent_vertices(v, graph);
		}
		InEdgeRange InEdges(const Vertex& v) const
		{
			return in_edges(v, graph);
		}
		OutEdgeRange OutEdges(const Vertex& v) const
		{
			return out_edges(v, graph);
		}
		VerticesSize NumVertices() const
		{
			return num_vertices(graph);
		}
		EdgesSize NumEdges() const
		{
			return num_edges(graph);
		}
		DegreeSize InDegree(const Vertex& v) const
		{
			return in_degree(v, graph);
		}
		DegreeSize OutDegree(const Vertex& v) const
		{
			return out_degree(v, graph);
		}
		bool HasEdge(const Vertex& src, const Vertex& tgt) const
		{
			std::pair<Edge, bool> result = edge(src, tgt, graph);
			return result.second;
		}
		Vertex Source(const Edge& e) const
		{
			return source(e, graph);
		}
		Vertex Target(const Edge& e) const
		{
			return target(e, graph);
		}

		/* operators */
		Graph& operator=(const Graph &rhs)
		{
			graph = rhs.graph;
			return *this;
		}

	protected:
		GraphContainer graph;
	};

} // end of namespace utils
} // end of namespace minerva

#endif
