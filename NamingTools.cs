/* (c) ALTIMESH 2018 -- all rights reserved */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

#pragma warning disable 1591
namespace Altimesh.Hybridizer.Runtime
{
    public interface ITypeIdProvider
    {
        int GetTypeID(Type t);
    }

    /// <summary>
    /// 
    /// </summary>
    public class NamingTools
    {
        private static int _heapArrayAllocationCount = 0;
        /// <summary>
        /// internal
        /// </summary>
        /// <returns></returns>
        public static string GetHeapArrayAllocationName()
        {
            return String.Format("__hyb__allocation{0}", _heapArrayAllocationCount++);
        }

        /// <summary>
        /// mangling of typename
        /// </summary>
        /// <param name="t"></param>
        /// <param name="nonamespace"></param>
        /// <returns></returns>
        public static string CSharpTypeName(Type t, out bool nonamespace)
        {
            //   error CS0673: Impossible d'utiliser System.Void dans C# : utilisez typeof(void) pour obtenir l'objet de type void
            if (t == typeof(void)) { nonamespace = true; return "void"; }


            nonamespace = false;
            if (t.IsArray) return CSharpTypeName(t.GetElementType(), out nonamespace) + "[]";
            if (t.IsByRef) return "ref " + CSharpTypeName(t.GetElementType(), out nonamespace);
            if (t.IsGenericParameter) return t.Name;
            if (!t.IsGenericType)
            {
                if (t.IsNested)
                    return CSharpTypeName(t.DeclaringType, out nonamespace) + "." + t.Name;
                return t.Name;
            }
            // if (t.IsGenericTypeDefinition) return t.FullName.Substring(0, t.FullName.IndexOf('`')) + "<>" ;

            StringBuilder sb = new StringBuilder();
            int tick = t.GetGenericTypeDefinition().Name.IndexOf("`");
            string head = "";
            if (t.IsNested)
                head = CSharpTypeName(t.DeclaringType, out nonamespace) + ".";
            sb.AppendFormat("{0}{1}<", head, t.GetGenericTypeDefinition().Name.Substring(0, tick));
            bool first = true;
            foreach (Type tt in t.GetGenericArguments())
            {
                if (first) first = false; else sb.Append(",");
                sb.Append(CSharpTypeName(tt, out nonamespace));
            }
            sb.Append(">");
            return sb.ToString();
        }

        /// <summary>
        /// mangling of typename
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static string QualifiedTypeName(Type t, bool SimplifyGenerics = true)
        {
            if (t.IsPointer) return QualifiedTypeName(t.GetElementType()) + "*"; 
            if (t.IsArray)
            {
                int rank = t.GetArrayRank();
                return QualifiedTypeName(t.GetElementType()) + "[" + string.Join(",", Enumerable.Repeat("", rank).ToArray()) + "]";
            }
            if (t.IsByRef) return "ref " + QualifiedTypeName(t.GetElementType());
            if (t.IsGenericParameter) return t.Name;
            if (!t.IsGenericType)
            {
                return t.FullName;
            }
            // if (t.IsGenericTypeDefinition) return t.FullName.Substring(0, t.FullName.IndexOf('`')) + "<>" ;

            StringBuilder sb = new StringBuilder();
            int tick = t.GetGenericTypeDefinition().Name.IndexOf("`");
            if (tick == -1)
                tick = t.GetGenericTypeDefinition().Name.Length;
            string head = t.Namespace;
            if (t.IsNested)
                head = QualifiedTypeName(t.DeclaringType) + "+";
            else
                head = head + ".";
            sb.AppendFormat("{0}{1}<", head, t.GetGenericTypeDefinition().Name.Substring(0, tick));
            bool first = true;
			int count = 0;
            foreach (Type tt in t.GetGenericArguments())
            {
                if (first) first = false; else sb.Append(",");
				if (SimplifyGenerics)
				{
					if (tt.IsGenericParameter)
						sb.Append("T" + count++);
					else
						sb.Append(QualifiedTypeName(tt));
				}
				else
				{
					sb.Append(QualifiedTypeName(tt));
				}
            }
            sb.Append(">");
            return sb.ToString();
        }

        class ParsedType
        {
            public ParsedType outerType;
            public string name;
            public List<ParsedType> genericArguments = new List<ParsedType>();

            private void output(StringBuilder sb, ref bool gen)
            {
                if (outerType != null)
                {
                    outerType.output(sb, ref gen);
                    sb.Append('+');
                }
                sb.Append(name);
                if (!gen && genericArguments.Any())
                {
                    sb.Append("`" + genericArguments.Count);
                    gen = true;
                }
            }

            public override string ToString()
            {
                bool gen = false;
                var sb = new StringBuilder();
                output(sb, ref gen);
                return sb.ToString();
            }

            public Type ToType(Type genericHolderType)
            {
                string fullGenericDefinition = ToString();
                Type holderType = QualifiedType(fullGenericDefinition, genericHolderType);
                List<Type> argumentList = genericArguments.Select(x => x.ToType(genericHolderType)).ToList();
                if (argumentList.Count == 0 || argumentList.Any(x => x == null)) 
                    return holderType;
                Type genType = Type.GetType(fullGenericDefinition, false);
                if (genType == null) return null;
                return genType.MakeGenericType(argumentList.ToArray());                
            }
        }

        static int findMatchingGt(string name, int startFrom)
        {
            int matchingGt = startFrom;
            int count = 1;
            for(; count > 0 && matchingGt < name.Length; matchingGt++)
            {
                if (name[matchingGt] == '>') count--;
                if (name[matchingGt] == '<') count++;
            }
            if (count == 0)
                return matchingGt - 1;
            throw new ApplicationException("Cannot parse type name " + name);
        }

        /// <summary>
        /// list is a type list expressed as coma separated list of types - NOTE: types may contain comas !
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        static string[] SplitTypeList(string list)
        {
            List<string> result = new List<string>();
            StringBuilder current = new StringBuilder();
            int nestingTemplate = 0;
            foreach (char c in list)
            {
                if (c == ',')
                {
                    if (nestingTemplate == 0)
                    {
                        result.Add(current.ToString());
                        current = new StringBuilder();
                        continue;
                    }
                }
                if (c == '<') ++nestingTemplate;
                if (c == '>') --nestingTemplate;
                current.Append(c);
            }
            if (nestingTemplate != 0) throw new ApplicationException("Cannot parse type name list " + list);
            result.Add(current.ToString());
            return result.ToArray();
        }

        static ParsedType parseGenericType(string typeName, ParsedType outer = null)
        {
            ParsedType res = new ParsedType();
            res.outerType = outer;
            var firstPlus = typeName.IndexOf('+');
            var firstLt = typeName.IndexOf("<");

            if (firstPlus < 0 && firstLt < 0)
            {
                res.name = typeName;
                return res;
            }

            var idx = 0;
            if (firstLt > 0 && (firstPlus == -1 || firstPlus > firstLt))
            {
                var gt = findMatchingGt(typeName, firstLt + 1);
                var genArgs = typeName.Substring(firstLt + 1, gt - firstLt - 1);
                string[] typeList = SplitTypeList(genArgs);
                res.genericArguments = typeList.Select(x => parseGenericType(x, null)).ToList();
                res.name = typeName.Substring(0, firstLt);
                idx = gt + 1;
            } 
            else if (firstPlus > 0)
            {
                res.name = typeName.Substring(0, firstPlus);
                idx = firstPlus;
            }
            else
            {
                res.name = typeName;
                idx = typeName.Length;
            }

            if (idx < typeName.Length && typeName[idx] == '+')
            {
                return parseGenericType(typeName.Substring(idx + 1), res);
            }
            
            return res;
        }

        private static string UnGenericize(string typeName, bool nested = false)
        {
            var parsed = parseGenericType(typeName, null);
            return parsed.ToString();
        }

        /// <summary>
        /// type from mangled typename
        /// </summary>
        /// <returns></returns>
        public static Type QualifiedType(string name, Type genericHolderType, Assembly asm = null)
        {
            if (name.EndsWith("*")) return QualifiedType(name.Substring(0, name.Length - 1), genericHolderType).MakePointerType();
            if (name.EndsWith("[]"))
            {
                var eltType = QualifiedType(name.Substring(0, name.Length - 2), genericHolderType);
                return eltType.MakeArrayType();
            }
            if (name.StartsWith("ref ")) return QualifiedType(name.Substring("ref ".Length), genericHolderType).MakeByRefType();
            if (name.StartsWith("<"))
            {
                if (genericHolderType == null) return null;
                foreach (Type t in genericHolderType.GetGenericArguments())
                {
                    if (string.Format("<{0}>", t.Name).Equals(name)) return t;
                }
            }

            if (name.Contains("<"))
            {
                var parsed = parseGenericType(name, null);
                return parsed.ToType(genericHolderType);
            }
            else
            {
                if (genericHolderType != null)
                {
                    // try types within generic arguments...
                    foreach (Type t in genericHolderType.GetGenericArguments())
                    {
                        if (t.Name == name)
                            return t;
                    }
                }

                Type res = Type.GetType(name, false);
                if(res == null && asm != null)
                    res = asm.GetType(name, false);
                return res;
            }
        }

        private static List<Type> ParseTypeList(string typelist, Type genericHolderType)
        {
            List<Type> result = new List<Type>();
            int nested = 0;
            string inner = typelist.Substring(1, typelist.Length - 2);
            string current = string.Empty;
            foreach (char c in inner)
            {
                switch (c)
                {
                    case '<': ++nested; break;
                    case '>': --nested; break;
                    case ',':
                        if (nested == 0)
                        {
                            result.Add(QualifiedType(current, genericHolderType));
                            current = string.Empty;
                        }
                        break;
                    default: current += c; break;
                }
            }
            if (current != string.Empty)
                result.Add(QualifiedType(current, genericHolderType));
            return result;
        }

        // For function pointers
        #region For function pointers

        /// <summary>
        /// mangling of char
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public static string Mangle(char c)
        {
            if ((c >= 'A') && (c <= 'Z')) return "" + c;
            if ((c >= 'a') && (c <= 'z')) return "" + c;
            if ((c >= '0') && (c <= '9')) return "" + c;
            if (c == '_') return "" + c;
            return "x" + ((int)c).ToString("D2");
        }

        /// <summary>
        /// mangling of typename
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static string GetFullTypeName(Type t)
        {
            if (t == typeof(double)) return "double";
            if (t == typeof(float)) return "float";
            if (t == typeof(int)) return "int";
            if (t == typeof(uint)) return "unsigned";
            if (t.IsGenericType)
            {
                if (t.IsGenericTypeDefinition)
                {
                    return t.FullName.Substring(0, t.FullName.IndexOf('`')) + "<>";
                }
                else
                {
                    StringBuilder sb = new StringBuilder();
                    string gtd = t.GetGenericTypeDefinition().FullName;
                    sb.Append(gtd.Substring(0, gtd.IndexOf('`')));
                    sb.Append("<");
                    bool first = true;
                    foreach (Type typename in t.GetGenericArguments())
                    {
                        if (first) { first = false; } else { sb.Append(", "); }
                        sb.Append(GetFullTypeName(typename));
                    }
                    sb.Append(">");
                    return sb.ToString();
                }
            }
            return t.FullName;
        }

        /// <summary>
        /// mangling of method name
        /// </summary>
        /// <param name="method"></param>
        /// <returns></returns>
        public static string GetFullMethodName(MethodBase method)
        {
            if (method.DeclaringType == null)
                return method.Name;
            else
                return GetFullTypeName(method.DeclaringType) + "." + method.Name;
        }

        /// <summary>
        /// mangling of method name
        /// </summary>
        /// <param name="method"></param>
        /// <returns></returns>
        public static string Mangle(MethodBase method)
        {
            return RawMangle(GetFullMethodName(method));
        }

        /// <summary>
        /// mangling of method name
        /// </summary>
        /// <param name="cmethodname"></param>
        /// <returns></returns>
        public static string RawMangle(string cmethodname)
        {
            StringBuilder result = new StringBuilder();
            foreach (char c in cmethodname)
            {
                result.Append(Mangle(c));
            }
            return result.ToString();
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol(Action action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1>(Action<T1> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2>(Action<T1, T2> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3>(Action<T1, T2, T3> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4>(Action<T1, T2, T3, T4> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Get the C linkage name of a method marked as [EntryPoint]
        /// </summary>
        /// <returns></returns>
        public static string GetEntryPointSymbol<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> action)
        {
            return RawMangle(GetFullMethodName(action.Method));
        }

        /// <summary>
        /// Short encoding of parameter types -- for mangling
        /// </summary>
        /// <param name="types"></param>
        /// <returns></returns>
        public static string EncodeParameterTypes(IEnumerable<Type> types)
        {
            StringBuilder sb = new StringBuilder();
            foreach (Type t in types)
            {
                sb.Append(EncodeParameterType(t));
            }
            return sb.ToString();
        }

        private static string EncodeParameterType(Type t)
        {
            if (t == typeof(double)) return "D";
            if (t == typeof(float)) return "F";
            if (t == typeof(int)) return "I";
            if (t == typeof(long)) return "L";
            if (t == typeof(short)) return "S";
            if (t.IsGenericParameter) return "T_" + RawMangle(t.Name);
            if (t.IsByRef) return EncodeParameterType(t.GetElementType()) + "r";
            if (t.IsPointer) return EncodeParameterType(t.GetElementType()) + "p";
            if (t.IsArray) return EncodeParameterType(t.GetElementType()) + "a";
            if ((t.IsGenericType) && (!t.IsGenericTypeDefinition))
            {
                // treatment for generics
                return "G_" + EncodeParameterType(t.GetGenericTypeDefinition()) + "_" + EncodeParameterTypes(t.GetGenericArguments()) + "_H";
            }
            return "X" + RawMangle(GetFullTypeName(t));
        }

        /// <summary>
        /// manging of method signature
        /// </summary>
        /// <param name="Original"></param>
        /// <returns></returns>
        public static string GetEncodedSignature(MethodBase Original)
        {
            var t = Original.DeclaringType;
            string res = string.Format("{0}_{1}_{2}", 
                RawMangle(NamingTools.QualifiedTypeName(t)), 
                RawMangle(Original.Name), 
                EncodeParameterTypes(Original.GetParameters().Select(x => x.ParameterType)));
            return res;
        }
        #endregion
    }

    public static class IdentifierMangler
    {
        public static HashSet<string> cppKeywords = new HashSet<string> {"alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr", "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "import", "inline", "int", "long", "module", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr", "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "override", "final", "audit", "axiom", "transaction_safe", "transaction_safe_dynamic" };

        //http://en.cppreference.com/w/cpp/language/identifiers
        public static string Mangle(string input)
        {
            if (String.IsNullOrEmpty(input))
            {
                throw new ApplicationException("empty lambda name");
            }

            if (cppKeywords.Contains(input))
            {
                return "_" + input;
            }

            StringBuilder sb = new StringBuilder();
            if (!Char.IsLetter(input.First()) && input.First() != '_')
            {
                sb.Append("_");
            }
            foreach (char s in input)
            {
                if (!Char.IsDigit(s) && !Char.IsLetter(s) && s != '_')
                {
                    sb.Append("x" + ((int)s).ToString("D2"));
                }
                else
                {
                    sb.Append(s);
                }
            }

            return sb.ToString();
        }
    }

    public static class CMethodMangler
    {
        public static string Mangle(char c)
        {
            if ((c >= 'A') && (c <= 'Z')) return "" + c;
            if ((c >= 'a') && (c <= 'z')) return "" + c;
            if ((c >= '0') && (c <= '9')) return "" + c;
            if (c == '_') return "" + c;
            return "x" + ((int)c).ToString("D2");
        }

        public static string GetFullMethodName(MethodBase method)
        {
            if (method.DeclaringType == null)
                return method.Name;
            else
                return method.DeclaringType.FullName + "." + method.Name;
        }

        public static string Mangle(MethodBase method)
        {
            return RawMangle(GetFullMethodName(method));
        }

        public static string RawMangle(string cmethodname)
        {
            StringBuilder result = new StringBuilder();
            foreach (char c in cmethodname)
            {
                result.Append(Mangle(c));
            }
            return result.ToString();
        }

        /// <summary>
        /// Short encoding of parameter types -- for mangling
        /// </summary>
        /// <param name="types"></param>
        /// <param name="typeIdProvider"></param>
        /// <returns></returns>
        public static string EncodeParameterTypes(Type[] types, ITypeIdProvider typeIdProvider)
        {
            StringBuilder sb = new StringBuilder();
            foreach (Type t in types)
            {
                sb.Append(EncodeParameterType(t, typeIdProvider));
            }
            return sb.ToString();
        }

        private static string EncodeParameterType(Type t, ITypeIdProvider typeIdProvider)
        {
            if (t == typeof(double)) return "D";
            if (t == typeof(float)) return "F";
            if (t == typeof(int)) return "I";
            if (t == typeof(long)) return "L";
            if (t == typeof(short)) return "S";
            if ((t.IsGenericType) && (!t.IsGenericTypeDefinition))
            {
                // treatment for generics
                return "G_" + EncodeParameterType(t.GetGenericTypeDefinition(), typeIdProvider) + "_" + EncodeParameterTypes(t.GetGenericArguments(), typeIdProvider) + "_H";
            }
            return "X" + typeIdProvider.GetTypeID(t);
        }

    }
}
#pragma warning restore 1591